"""Tests for the deduplicator module."""

import pytest
import polars as pl
from datetime import datetime, timedelta

from src.data.deduplicator import Deduplicator


@pytest.fixture
def sample_activities():
    """Create sample activities DataFrame for testing."""
    base_date = datetime(2024, 1, 1)
    
    return pl.DataFrame({
        "id": ["act1", "act2", "act1", "act3", "act2"],  # Duplicates
        "start_date_local": [
            base_date,
            base_date + timedelta(days=1),
            base_date,  # Duplicate of act1
            base_date + timedelta(days=2),
            base_date + timedelta(days=1),  # Duplicate of act2
        ],
        "name": ["Ride 1", "Ride 2", "Ride 1", "Ride 3", "Ride 2"],
        "distance": [20000, 30000, 20000, 25000, 30000],
        "moving_time": [3600, 5400, 3600, 4500, 5400],
        "file_source": ["file1.csv", "file1.csv", "file2.csv", "file1.csv", "file3.csv"],
        "import_timestamp": [
            datetime(2024, 1, 10, 10, 0),
            datetime(2024, 1, 10, 10, 0),
            datetime(2024, 1, 11, 10, 0),  # Later import
            datetime(2024, 1, 10, 10, 0),
            datetime(2024, 1, 12, 10, 0),  # Latest import
        ],
    })


def test_identify_exact_duplicates(sample_activities):
    """Test exact duplicate identification."""
    dedup = Deduplicator()
    
    df_with_flags = dedup.identify_exact_duplicates(sample_activities)
    
    # Check that duplicates are identified
    assert "is_exact_duplicate" in df_with_flags.columns
    assert "duplicate_count" in df_with_flags.columns
    
    # Count duplicates
    duplicate_count = df_with_flags.filter(pl.col("is_exact_duplicate"))["id"].n_unique()
    assert duplicate_count == 2  # act1 and act2 have duplicates


def test_deduplicate_keeps_latest(sample_activities):
    """Test that deduplication keeps the latest version by default."""
    dedup = Deduplicator()
    
    df_dedup, report = dedup.deduplicate(sample_activities)
    
    # Should keep 3 unique activities
    assert len(df_dedup) == 3
    assert df_dedup["id"].n_unique() == 3
    
    # Check that latest versions are kept
    act2_records = df_dedup.filter(pl.col("id") == "act2")
    assert len(act2_records) == 1
    assert act2_records["file_source"][0] == "file3.csv"  # Latest import


def test_deduplicate_report(sample_activities):
    """Test deduplication report generation."""
    dedup = Deduplicator()
    
    df_dedup, report = dedup.deduplicate(sample_activities)
    
    assert report["total_records"] == 5
    assert report["unique_records"] == 3
    assert report["exact_duplicates"] == 2
    assert "deduplication_time" in report


def test_track_file_sources(sample_activities):
    """Test that file sources are tracked correctly."""
    dedup = Deduplicator()
    
    df_with_sources = dedup.track_file_sources(sample_activities)
    
    assert "all_file_sources" in df_with_sources.columns
    
    # Check act1 has both file sources
    act1_sources = df_with_sources.filter(pl.col("id") == "act1")["all_file_sources"][0]
    assert "file1.csv" in act1_sources
    assert "file2.csv" in act1_sources


def test_empty_dataframe():
    """Test handling of empty DataFrame."""
    dedup = Deduplicator()
    
    empty_df = pl.DataFrame()
    df_dedup, report = dedup.deduplicate(empty_df)
    
    assert df_dedup.is_empty()
    assert report["total_records"] == 0
    assert report["unique_records"] == 0


def test_no_duplicates():
    """Test handling of DataFrame with no duplicates."""
    dedup = Deduplicator()
    
    unique_df = pl.DataFrame({
        "id": ["act1", "act2", "act3"],
        "start_date_local": [datetime(2024, 1, i) for i in range(1, 4)],
        "name": [f"Ride {i}" for i in range(1, 4)],
        "file_source": ["file1.csv"] * 3,
        "import_timestamp": [datetime.now()] * 3,
    })
    
    df_dedup, report = dedup.deduplicate(unique_df)
    
    assert len(df_dedup) == 3
    assert report["exact_duplicates"] == 0
    assert report["unique_records"] == 3