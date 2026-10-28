"""Tests for profile_service.py"""

from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from core.services.profile_service import delete_profile


@pytest.mark.unit
class TestDeleteProfile:
    """delete_profile drops the profile's vector-store collection along with its rows."""

    @pytest.fixture
    def events(self):
        """Order in which the vector store and the session were called."""
        return []

    @pytest.fixture
    def mock_db_session(self, events):
        """Create a mock async database session with no related reviews or sources."""
        session = AsyncMock()
        result = Mock()
        result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=result)
        session.delete = AsyncMock(side_effect=lambda obj: events.append("db.delete"))
        session.commit = AsyncMock(side_effect=lambda: events.append("db.commit"))
        session.rollback = AsyncMock()
        return session

    @pytest.fixture
    def mock_vector_store(self, events):
        """Create a mock vector store that records when its collection is dropped."""
        store = Mock()
        store.delete_collection = Mock(side_effect=lambda name: events.append("drop"))
        return store

    @pytest.fixture
    def mock_profile(self):
        """Create a mock Profile object."""
        profile = Mock()
        profile.id = uuid4()
        profile.user_id = uuid4()
        return profile

    @pytest.mark.asyncio
    async def test_drops_profile_collection_before_deleting_rows(
        self, mock_db_session, mock_vector_store, mock_profile, events
    ):
        """The profile's collection is dropped before any row is deleted or committed."""
        with patch(
            "core.services.profile_service.get_profile", AsyncMock(return_value=mock_profile)
        ):
            result = await delete_profile(
                mock_db_session, mock_profile.id, mock_profile.user_id, mock_vector_store
            )

        assert result is True
        mock_vector_store.delete_collection.assert_called_once_with(f"profile_{mock_profile.id}")
        assert events == ["drop", "db.delete", "db.commit"]

    @pytest.mark.asyncio
    async def test_vector_store_failure_leaves_rows_untouched(
        self, mock_db_session, mock_vector_store, mock_profile
    ):
        """A vector-store failure rolls back and raises before any row is deleted."""
        mock_vector_store.delete_collection.side_effect = RuntimeError("chroma unavailable")

        with (
            patch(
                "core.services.profile_service.get_profile",
                AsyncMock(return_value=mock_profile),
            ),
            pytest.raises(RuntimeError, match="chroma unavailable"),
        ):
            await delete_profile(
                mock_db_session, mock_profile.id, mock_profile.user_id, mock_vector_store
            )

        mock_db_session.delete.assert_not_awaited()
        mock_db_session.commit.assert_not_awaited()
        mock_db_session.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_missing_profile_never_touches_vector_store(
        self, mock_db_session, mock_vector_store
    ):
        """A profile that is missing or not owned returns False without a vector call."""
        with patch("core.services.profile_service.get_profile", AsyncMock(return_value=None)):
            result = await delete_profile(mock_db_session, uuid4(), uuid4(), mock_vector_store)

        assert result is False
        mock_vector_store.delete_collection.assert_not_called()
