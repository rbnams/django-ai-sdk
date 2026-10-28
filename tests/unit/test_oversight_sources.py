"""What retrieval found reaches oversight: linked memories become tools, earlier turns stay apart."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from django_ai_sdk import Agent
from django_ai_sdk.adapters.citations import NumberedSource, SourceRegistry
from django_ai_sdk.conversation.models import Thread
from django_ai_sdk.memories.models import Entry, ThreadMemory
from django_ai_sdk.oversight.session import TurnSession
from django_ai_sdk.protocols.vercel import VercelProtocolHandler
from django_ai_sdk.storage.memory import MemoryStorageAdapter
from tests.factories.db import MemoryFactory, UserFactory
from tests.mocks.permissions import memory_permissions
from tests.unit.test_oversight_turns import make_agent


class HandbookAgent(Agent):
    name = "linked_rag_test"
    model = "gpt-4o-mini"
    instructions = ["Answer from the handbook."]
    protocol = VercelProtocolHandler
    storage_adapter = MemoryStorageAdapter
    memories = ["test-handbook"]  # noqa: RUF012

    async def get_pipeline_adapter(self, thread_id=None, user=None):
        raise NotImplementedError


@pytest.mark.django_db
class TestLinkedRagTools:
    @pytest.fixture(autouse=True)
    def _memories_open(self):
        with memory_permissions("django_ai_sdk.permissions.AllowAll"):
            yield

    async def test_the_agents_memories_are_linked_and_searched(self):
        memory = await MemoryFactory.acreate(name="Test handbook", slug="test-handbook")
        await Entry.objects.acreate(memory=memory, content="Staff get 25 days of leave.")
        thread = await Thread.objects.acreate()
        user = await UserFactory.acreate()
        agent = HandbookAgent()
        tool = MagicMock()
        agent.rag_provider = MagicMock(get_tool=AsyncMock(return_value=tool))
        registry = SourceRegistry()

        tools = await agent.get_linked_rag_tools(str(thread.id), user, citation_registry=registry)

        assert tools == [tool]
        assert await ThreadMemory.objects.filter(thread=thread, memory=memory).aexists()
        # The turn's registry goes with the tool, so what it finds reaches oversight.
        assert agent.rag_provider.get_tool.await_args.kwargs["citation_registry"] is registry

    async def test_nothing_without_a_provider_memories_or_a_signed_in_user(self):
        thread = await Thread.objects.acreate()
        user = await UserFactory.acreate()
        agent = HandbookAgent()
        assert await agent.get_linked_rag_tools(str(thread.id), user) == []  # no provider
        agent.rag_provider = MagicMock(get_tool=AsyncMock())
        assert await agent.get_linked_rag_tools(str(thread.id), None) == []
        agent.memories = []
        assert await agent.get_linked_rag_tools(str(thread.id), user) == []
        agent.rag_provider.get_tool.assert_not_called()


@pytest.mark.django_db
class TestSeededSources:
    async def test_sources_from_earlier_turns_are_not_counted_as_found(self):
        registry = SourceRegistry()
        earlier = NumberedSource(title="Leave · §1", content="25 days", doc_id="d1", chunk_id="c1")
        registry.seed([earlier])
        registry.add(
            [NumberedSource(title="Pay · §2", content="the 25th", doc_id="d2", chunk_id="c2")]
        )
        session = await TurnSession.start(make_agent([], None), channel="api", input="q")

        session.set_sources(registry.all, seeded=registry.seeded)

        assert [s.title for s in session.view.sources] == ["Leave · §1", "Pay · §2"]
        assert [s.title for s in session.view.fresh_sources()] == ["Pay · §2"]


@pytest.mark.django_db
class TestChunkTitles:
    async def test_every_chunk_carries_its_documents_name_to_the_embedder(self):
        from django_ai_sdk.rags.qdrant_hybrid import QdrantBM25HybridRAGConfig

        memory = await MemoryFactory.acreate(name="CAOs")
        entry = await Entry.objects.acreate(memory=memory, name="CAO VVT.pdf", content="2% loonsom")
        assert entry.to_rag_document().metadata["title"] == "CAO VVT.pdf"
        assert "title" in QdrantBM25HybridRAGConfig().meta_fields_to_embed
