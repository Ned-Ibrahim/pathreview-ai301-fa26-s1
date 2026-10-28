"""Tests for vector_store.py"""

import pytest

from rag.retriever.vector_store import VectorStore


def collection_names(store: VectorStore) -> list[str]:
    """Names of the store's collections (list_collections returns names or objects by version)."""
    return [getattr(c, "name", c) for c in store.client.list_collections()]


@pytest.mark.unit
class TestDeleteCollection:
    """VectorStore.delete_collection removes a collection and tolerates a missing one."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a VectorStore persisted in a temporary directory."""
        return VectorStore(persist_dir=str(tmp_path))

    def test_deletes_existing_collection_and_its_chunks(self, store):
        """An existing collection and its chunks are gone afterwards."""
        collection = store.get_collection("profile_abc")
        collection.add(ids=["chunk_0"], embeddings=[[0.1, 0.2, 0.3]], documents=["resume text"])

        store.delete_collection("profile_abc")

        assert "profile_abc" not in collection_names(store)

    def test_missing_collection_is_a_no_op(self, store):
        """Deleting a collection that does not exist neither raises nor creates it."""
        store.delete_collection("profile_missing")

        assert "profile_missing" not in collection_names(store)

    def test_other_collections_are_untouched(self, store):
        """Only the named collection is deleted."""
        store.get_collection("profile_abc")
        store.get_collection("profile_other")

        store.delete_collection("profile_abc")

        assert collection_names(store) == ["profile_other"]
