"""The multi-query retrievers are haystack components: they must construct."""

from __future__ import annotations

from haystack.document_stores.in_memory import InMemoryDocumentStore

from django_ai_sdk.rags.components import MultiQueryBM25Retriever


def test_the_bm25_retriever_constructs():
    # haystack's @component re-creates the class, so a zero-argument super() in
    # __init__ points at a class the instance is not an instance of.
    retriever = MultiQueryBM25Retriever(document_store=InMemoryDocumentStore(), top_k=2)
    assert retriever.top_k == 2
