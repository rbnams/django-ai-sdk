"""DJA-98: a document saved in another process (a task worker) reaches the chat's index.

The worker never serves a chat, so it has no cached RAG. It writes to the memory's
index itself when that index lives on a server, which every process shares. Only the
provider's choice is tested here: the RAG is a small fake with a real storage config,
so no Qdrant server or embedding model is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from django_ai_sdk.rags.config import QdrantStorageConfig
from django_ai_sdk.rags.provider import RAGProvider
from django_ai_sdk.rags.schemas import RagDocument

SERVER = QdrantStorageConfig(backend="server", location="http://qdrant:6333")
LOCAL = QdrantStorageConfig(backend="persistent", location="/tmp/qdrant")


class FakeRag:
    needs_warmup = True

    def __init__(self, storage: QdrantStorageConfig) -> None:
        self.config = SimpleNamespace(storage=storage)
        self.warmups = 0
        self.added: list[RagDocument] = []
        self.removed: list[str] = []

    async def warmup(self, force_rebuild: bool = False) -> None:
        self.warmups += 1

    async def add_documents(self, documents: list[RagDocument]) -> None:
        self.added += documents

    async def remove_documents(self, document_ids: list[str]) -> None:
        self.removed += document_ids


class FakeAgent:
    def __init__(self, rag: FakeRag | None) -> None:
        self.rag = rag
        self.builds = 0

    async def get_rag_pipeline(self, memory_id: str | None = None) -> FakeRag | None:
        self.builds += 1
        return self.rag


DOC = RagDocument(id="e1", content="XAF 4.0 explained")


@pytest.mark.asyncio
class TestSharedIndex:
    async def test_worker_writes_to_the_server_index(self):
        provider, rag = RAGProvider(), FakeRag(SERVER)

        assert await provider.add_documents(FakeAgent(rag), "m1", [DOC])

        assert rag.added == [DOC]
        assert rag.warmups == 1  # opens the existing collection, nothing rebuilt
        assert provider._cache == {}  # the worker keeps no warm index

    async def test_worker_removes_from_the_server_index(self):
        provider, rag = RAGProvider(), FakeRag(SERVER)

        assert await provider.remove_documents(FakeAgent(rag), "m1", ["e1"])

        assert rag.removed == ["e1"]

    async def test_a_per_process_index_is_left_alone(self):
        # A local folder or in-memory index belongs to the process that built it.
        provider, rag = RAGProvider(), FakeRag(LOCAL)

        assert not await provider.add_documents(FakeAgent(rag), "m1", [DOC])
        assert not await provider.remove_documents(FakeAgent(rag), "m1", ["e1"])

        assert (rag.added, rag.removed, rag.warmups) == ([], [], 0)

    async def test_the_cached_index_is_written_without_building_one(self):
        provider, cached = RAGProvider(), FakeRag(LOCAL)
        agent = FakeAgent(FakeRag(SERVER))
        provider._cache[provider._get_cache_key(agent, "m1")] = cached

        assert await provider.add_documents(agent, "m1", [DOC])

        assert cached.added == [DOC]
        assert agent.builds == 0

    async def test_a_memory_without_documents_gets_its_index_later(self):
        # The first chat message came before its first document was processed.
        provider, agent = RAGProvider(), FakeAgent(None)

        assert await provider.get_rag_instance(agent, "m1") is None
        assert provider._cache == {}  # not remembered as "no index"

        agent.rag = FakeRag(SERVER)
        assert await provider.get_rag_instance(agent, "m1") is agent.rag

    async def test_agents_of_different_classes_write_a_shared_index_once(self):
        # Every agent class with a provider is told of a save; a server index is one.
        class Other(FakeAgent):
            pass

        provider, rag = RAGProvider(), FakeRag(SERVER)
        written: set[str] = set()

        for agent in (FakeAgent(rag), Other(rag)):
            assert await provider.add_documents(agent, "m1", [DOC], written=written)
        for agent in (FakeAgent(rag), Other(rag)):
            assert await provider.remove_documents(agent, "m1", ["e1"], written=set())

        assert rag.added == [DOC]
        assert rag.removed == ["e1", "e1"]  # each call had its own `written`
        assert written == {"http://qdrant:6333/default"}


@pytest.mark.asyncio
async def test_an_edited_document_replaces_its_old_chunks():
    from unittest.mock import AsyncMock, MagicMock, patch

    from django_ai_sdk.rags.qdrant_hybrid import QdrantBM25HybridRAG

    rag = QdrantBM25HybridRAG(documents=[])
    rag._cached_document_store = MagicMock()
    calls: list[str] = []
    with (
        patch.object(
            rag,
            "remove_documents",
            AsyncMock(side_effect=lambda ids: calls.append(f"remove {ids}")),
        ),
        patch.object(
            rag,
            "_index_documents",
            AsyncMock(side_effect=lambda docs, store: calls.append("index")),
        ),
    ):
        await rag.add_documents([DOC])

    assert calls == ["remove ['e1']", "index"]
