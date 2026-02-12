"""Tests for database manager module."""

import pytest
import polars as pl
from datetime import datetime
import tempfile
from pathlib import Path

from src.storage.database.manager import DatabaseManager


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    db = DatabaseManager(db_path=db_path)
    yield db
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def sample_activities():
    """Create sample activities for testing."""
    return pl.DataFrame({
        "id": ["act1", "act2", "act3"],
        "start_date_local": [datetime(2024, 1, i) for i in range(1, 4)],
        "name": [f"Ride {i}" for i in range(1, 4)],
        "type": ["Ride"] * 3,
        "moving_time": [3600, 5400, 4500],
        "distance": [30000, 45000, 35000],
        "total_elevation_gain": [300, 500, 400],
        "average_speed": [8.33, 8.33, 7.78],
        "file_source": ["test.csv"] * 3,
        "file_hash": ["hash1", "hash2", "hash3"],
        "import_timestamp": [datetime.now()] * 3,
    })


def test_database_initialization(temp_db):
    """Test database initialization and schema creation."""
    # Check that tables exist
    with temp_db.engine.connect() as conn:
        from sqlalchemy import text
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = [row[0] for row in result]
    
    assert "activities" in tables
    assert "ingestion_history" in tables
    assert "data_quality_metrics" in tables


def test_save_activities(temp_db, sample_activities):
    """Test saving activities to database."""
    stats = temp_db.save_activities(sample_activities)
    
    assert stats["saved"] == 3
    assert stats["updated"] == 0
    assert stats["skipped"] == 0
    
    # Save again - should update
    stats = temp_db.save_activities(sample_activities, update_existing=True)
    
    assert stats["saved"] == 0
    assert stats["updated"] == 3
    assert stats["skipped"] == 0


def test_get_activities(temp_db, sample_activities):
    """Test retrieving activities from database."""
    # Save activities first
    temp_db.save_activities(sample_activities)
    
    # Retrieve all
    activities = temp_db.get_activities()
    assert len(activities) == 3
    
    # Retrieve with date filter
    activities = temp_db.get_activities(
        start_date=datetime(2024, 1, 2),
        end_date=datetime(2024, 1, 3)
    )
    assert len(activities) == 2
    
    # Retrieve with type filter
    activities = temp_db.get_activities(activity_type="Ride")
    assert len(activities) == 3
    
    # Retrieve with limit
    activities = temp_db.get_activities(limit=2)
    assert len(activities) == 2


def test_save_ingestion_record(temp_db):
    """Test saving ingestion history."""
    metadata = {
        "file_path": "/path/to/file.csv",
        "file_name": "file.csv",
        "file_hash": "abc123",
        "file_size": 1024,
        "record_count": 100,
        "new_activities": 50,
        "duplicate_activities": 50,
    }
    
    temp_db.save_ingestion_record(metadata, status="success")
    
    # Retrieve history
    history = temp_db.get_ingestion_history(limit=1)
    
    assert len(history) == 1
    assert history["file_name"][0] == "file.csv"
    assert history["status"][0] == "success"


def test_get_summary_stats(temp_db, sample_activities):
    """Test getting summary statistics."""
    # Empty database
    stats = temp_db.get_summary_stats()
    assert stats["total_activities"] == 0
    
    # With activities
    temp_db.save_activities(sample_activities)
    stats = temp_db.get_summary_stats()
    
    assert stats["total_activities"] == 3
    assert stats["total_distance"] == 110000  # Sum of distances
    assert stats["total_moving_time"] == 13500  # Sum of moving times
    assert "earliest_activity" in stats
    assert "latest_activity" in stats
    assert stats["activities_by_type"]["Ride"] == 3


def test_column_filtering(temp_db):
    """Test that only valid columns are saved."""
    # DataFrame with extra columns
    df_with_extra = pl.DataFrame({
        "id": ["act1"],
        "start_date_local": [datetime.now()],
        "name": ["Test Ride"],
        "type": ["Ride"],
        "moving_time": [3600.0],
        "distance": [30000.0],
        "invalid_column": ["should_be_filtered"],
        "another_invalid": [123],
    })
    
    stats = temp_db.save_activities(df_with_extra)
    assert stats["saved"] == 1
    
    # Retrieve and check columns
    activities = temp_db.get_activities()
    assert "invalid_column" not in activities.columns
    assert "another_invalid" not in activities.columns
    assert "id" in activities.columns
    assert "name" in activities.columns


def test_empty_dataframe(temp_db):
    """Test handling of empty DataFrame."""
    empty_df = pl.DataFrame()
    
    stats = temp_db.save_activities(empty_df)
    assert stats["saved"] == 0
    assert stats["updated"] == 0
    assert stats["skipped"] == 0