"""Tests for Intervals.icu API integration."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import json
import polars as pl

from src.integrations.intervals_icu import IntervalsICUClient, IntervalsICUSyncManager


class TestIntervalsICUClient:
    """Tests for IntervalsICUClient."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return IntervalsICUClient("test_athlete", "test_api_key")
    
    def test_client_initialization(self):
        """Test client initialization."""
        client = IntervalsICUClient("athlete123", "key456")
        
        assert client.athlete_id == "athlete123"
        assert client.api_key == "key456"
        assert "Authorization" in client.session.headers
        assert client.session.headers["Authorization"] == "Bearer key456"
        assert client.max_calls_per_minute == 90
    
    @patch("src.integrations.intervals_icu.time.sleep")
    def test_rate_limiting(self, mock_sleep, client):
        """Test rate limiting mechanism."""
        import time
        
        # Set client to near rate limit
        client.rate_limit_calls = 89
        client.rate_limit_reset = time.time()
        
        # Next call should be allowed
        client._check_rate_limit()
        assert client.rate_limit_calls == 90
        
        # Call after limit should wait
        client._check_rate_limit()
        mock_sleep.assert_called_once()
        assert client.rate_limit_calls == 1
    
    @patch("requests.Session.request")
    def test_make_request_success(self, mock_request, client):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_request.return_value = mock_response
        
        result = client._make_request("GET", "/test")
        
        assert result == {"data": "test"}
        mock_request.assert_called_once()
    
    @patch("requests.Session.request")
    def test_make_request_auth_failure(self, mock_request, client):
        """Test authentication failure handling."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_request.return_value = mock_response
        
        result = client._make_request("GET", "/test")
        
        assert result is None
    
    @patch("requests.Session.request")
    @patch("src.integrations.intervals_icu.time.sleep")
    def test_make_request_rate_limit(self, mock_sleep, mock_request, client):
        """Test rate limit response handling."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "5"}
        mock_request.return_value = mock_response
        
        result = client._make_request("GET", "/test")
        
        # Should sleep for retry-after duration
        mock_sleep.assert_called_with(5)
    
    @patch("src.integrations.intervals_icu.IntervalsICUClient._make_request")
    def test_get_activities(self, mock_request, client):
        """Test getting activities."""
        mock_activities = [
            {
                "id": "act1",
                "start_date_local": "2024-01-15T10:00:00",
                "name": "Morning Ride",
                "icu_training_load": 50
            },
            {
                "id": "act2",
                "start_date_local": "2024-01-14T09:00:00",
                "name": "Strava Activity",
                "icu_sync_error": "Strava activity"
            }
        ]
        mock_request.return_value = mock_activities
        
        result = client.get_activities(
            oldest=datetime(2024, 1, 14),
            newest=datetime(2024, 1, 15)
        )
        
        # Should filter out Strava stub
        assert len(result) == 1
        assert result[0]["id"] == "act1"
        
        mock_request.assert_called_once_with(
            "GET",
            "/athlete/test_athlete/activities",
            params={
                "oldest": "2024-01-14",
                "newest": "2024-01-15"
            }
        )
    
    @patch("src.integrations.intervals_icu.IntervalsICUClient._make_request")
    def test_get_activities_default_dates(self, mock_request, client):
        """Test getting activities with default date range."""
        mock_request.return_value = []
        
        result = client.get_activities()
        
        # Should use 30 days ago as default
        call_args = mock_request.call_args
        assert "oldest" in call_args[1]["params"]
        # Check it's approximately 30 days ago
        oldest_date = datetime.strptime(call_args[1]["params"]["oldest"], "%Y-%m-%d")
        expected_date = datetime.now() - timedelta(days=30)
        assert abs((oldest_date - expected_date).days) <= 1
    
    @patch("src.integrations.intervals_icu.IntervalsICUClient.get_activities")
    def test_get_activities_since_last_import(self, mock_get_activities, client):
        """Test incremental sync since last import."""
        # Mock paginated responses with valid dates
        batch1 = []
        for i in range(100):
            day = 15 - (i % 15)  # Keep day in valid range 1-15
            batch1.append({"id": f"act{i}", "start_date_local": f"2024-01-{day:02d}T10:00:00"})
        
        batch2 = []
        for i in range(100, 150):
            day = 10 - (i % 10)  # Keep day in valid range 1-10
            batch2.append({"id": f"act{i}", "start_date_local": f"2024-01-{day:02d}T10:00:00"})
        
        mock_get_activities.side_effect = [batch1, batch2]
        
        last_import = datetime(2024, 1, 1)
        result = client.get_activities_since_last_import(last_import, batch_size=100)
        
        assert len(result) == 150
        assert mock_get_activities.call_count == 2
    
    def test_activities_to_dataframe(self, client):
        """Test converting activities to DataFrame."""
        activities = [
            {
                "id": "act1",
                "start_date_local": "2024-01-15T10:00:00",
                "name": "Morning Ride",
                "distance": 30000,
                "icu_training_load": 50
            },
            {
                "id": "act2",
                "start_date_local": "2024-01-14T09:00:00",
                "name": "Evening Ride",
                "distance": 25000,
                "icu_training_load": 45
            }
        ]
        
        df = client.activities_to_dataframe(activities)
        
        assert len(df) == 2
        assert "id" in df.columns
        assert "start_date_local" in df.columns
        assert df["start_date_local"].dtype == pl.Datetime
        # Should be sorted by date descending
        assert df["id"][0] == "act1"  # Most recent first
    
    def test_activities_to_dataframe_empty(self, client):
        """Test converting empty activities list."""
        df = client.activities_to_dataframe([])
        
        assert df.is_empty()
        assert isinstance(df, pl.DataFrame)


class TestIntervalsICUSyncManager:
    """Tests for IntervalsICUSyncManager."""
    
    @pytest.fixture
    def temp_state_file(self, tmp_path):
        """Create temporary state file."""
        return tmp_path / "sync_state.json"
    
    @pytest.fixture
    def mock_client(self):
        """Create mock client."""
        return Mock(spec=IntervalsICUClient)
    
    @pytest.fixture
    def sync_manager(self, mock_client, temp_state_file):
        """Create sync manager."""
        return IntervalsICUSyncManager(mock_client, temp_state_file)
    
    def test_load_state_new_file(self, sync_manager):
        """Test loading state when file doesn't exist."""
        assert sync_manager.state == {}
        assert sync_manager.get_last_sync_date() is None
    
    def test_load_state_existing_file(self, mock_client, temp_state_file):
        """Test loading existing state file."""
        # Create state file
        state_data = {
            "last_sync_date": "2024-01-15T10:00:00",
            "last_sync_count": 5
        }
        with open(temp_state_file, "w") as f:
            json.dump(state_data, f)
        
        sync_manager = IntervalsICUSyncManager(mock_client, temp_state_file)
        
        assert sync_manager.state == state_data
        assert sync_manager.get_last_sync_date() == datetime(2024, 1, 15, 10, 0, 0)
    
    def test_save_state(self, sync_manager, temp_state_file):
        """Test saving state to file."""
        sync_manager.state = {
            "last_sync_date": "2024-01-20T12:00:00",
            "last_sync_count": 10
        }
        sync_manager._save_state()
        
        # Read back and verify
        with open(temp_state_file, "r") as f:
            saved_state = json.load(f)
        
        assert saved_state == sync_manager.state
    
    def test_sync_activities_full(self, sync_manager, mock_client):
        """Test full sync of activities."""
        mock_activities = [
            {"id": "act1", "start_date_local": "2024-01-15T10:00:00"},
            {"id": "act2", "start_date_local": "2024-01-14T09:00:00"}
        ]
        mock_client.get_activities.return_value = mock_activities
        mock_client.activities_to_dataframe.return_value = pl.DataFrame(mock_activities)
        
        result = sync_manager.sync_activities(force_full=True)
        
        # Should call get_activities with 365 days
        mock_client.get_activities.assert_called_once()
        call_args = mock_client.get_activities.call_args
        oldest = call_args[1]["oldest"]
        expected = datetime.now() - timedelta(days=365)
        assert abs((oldest - expected).days) <= 1
        
        # Should update state
        assert sync_manager.state["last_sync_date"] == "2024-01-15T10:00:00"
        assert sync_manager.state["last_sync_count"] == 2
    
    def test_sync_activities_incremental(self, sync_manager, mock_client):
        """Test incremental sync."""
        # Set last sync date
        sync_manager.state = {"last_sync_date": "2024-01-10T10:00:00"}
        
        mock_activities = [
            {"id": "act3", "start_date_local": "2024-01-16T10:00:00"}
        ]
        mock_client.get_activities_since_last_import.return_value = mock_activities
        mock_client.activities_to_dataframe.return_value = pl.DataFrame(mock_activities)
        
        result = sync_manager.sync_activities()
        
        # Should call incremental sync
        mock_client.get_activities_since_last_import.assert_called_once_with(
            datetime(2024, 1, 10, 10, 0, 0)
        )
        
        # Should update state
        assert sync_manager.state["last_sync_date"] == "2024-01-16T10:00:00"
        assert sync_manager.state["last_sync_count"] == 1
    
    def test_sync_activities_no_new(self, sync_manager, mock_client):
        """Test sync when no new activities."""
        sync_manager.state = {"last_sync_date": "2024-01-10T10:00:00"}
        
        mock_client.get_activities_since_last_import.return_value = []
        mock_client.activities_to_dataframe.return_value = pl.DataFrame()
        
        result = sync_manager.sync_activities()
        
        assert result.is_empty()
        # State should be updated with timestamp but count should be 0
        assert sync_manager.state["last_sync_count"] == 0
        assert "last_sync_timestamp" in sync_manager.state
    
    def test_get_sync_status(self, sync_manager, mock_client):
        """Test getting sync status."""
        sync_manager.state = {
            "last_sync_date": "2024-01-15T10:00:00",
            "last_sync_timestamp": "2024-01-15T11:00:00",
            "last_sync_count": 5
        }
        mock_client.athlete_id = "test_athlete"
        
        status = sync_manager.get_sync_status()
        
        assert status["last_sync_date"] == "2024-01-15T10:00:00"
        assert status["last_sync_timestamp"] == "2024-01-15T11:00:00"
        assert status["last_sync_count"] == 5
        assert status["athlete_id"] == "test_athlete"