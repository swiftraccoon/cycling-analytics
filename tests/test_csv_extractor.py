"""Tests for CSV extractor module."""

import pytest
import polars as pl
from pathlib import Path
import hashlib
import tempfile
import shutil
from datetime import datetime

from src.etl.extractors.csv_extractor import CSVExtractor


@pytest.fixture
def temp_bronze_dir():
    """Create temporary bronze directory structure."""
    temp_dir = tempfile.mkdtemp(prefix="bronze_test_")
    bronze_path = Path(temp_dir)
    (bronze_path / "incoming").mkdir(parents=True)
    (bronze_path / "archive").mkdir(parents=True)
    
    yield bronze_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_csv_content():
    """Generate sample CSV content matching Intervals.icu format."""
    return """id,start_date_local,name,type,moving_time,distance,icu_average_watts,icu_ftp
i1001,2024-01-01,Morning Ride,Ride,3600,30000,200,250
i1002,2024-01-02,Tempo Ride,Ride,5400,45000,220,250
i1003,2024-01-03,Recovery Spin,Ride,2700,20000,150,250"""


def test_csv_extractor_initialization(temp_bronze_dir):
    """Test CSV extractor initialization."""
    extractor = CSVExtractor(bronze_path=str(temp_bronze_dir))
    
    assert extractor.bronze_path == temp_bronze_dir
    assert extractor.incoming_path == temp_bronze_dir / "incoming"
    assert extractor.archive_path == temp_bronze_dir / "archive"


def test_calculate_file_hash(temp_bronze_dir, sample_csv_content):
    """Test file hash calculation."""
    extractor = CSVExtractor(bronze_path=str(temp_bronze_dir))
    
    # Create test file
    test_file = temp_bronze_dir / "incoming" / "test.csv"
    test_file.write_text(sample_csv_content)
    
    # Calculate hash
    file_hash = extractor.calculate_file_hash(test_file)
    
    # Verify hash
    assert file_hash is not None
    assert len(file_hash) == 64  # SHA-256 produces 64 character hex string
    
    # Verify hash is consistent
    hash2 = extractor.calculate_file_hash(test_file)
    assert file_hash == hash2
    
    # Verify hash changes with content
    test_file.write_text(sample_csv_content + "\nextra_line")
    hash3 = extractor.calculate_file_hash(test_file)
    assert hash3 != file_hash


def test_extract_from_file(temp_bronze_dir, sample_csv_content):
    """Test extracting data from a single CSV file."""
    extractor = CSVExtractor(bronze_path=str(temp_bronze_dir))
    
    # Create test file
    test_file = temp_bronze_dir / "incoming" / "test.csv"
    test_file.write_text(sample_csv_content)
    
    # Extract data
    df, metadata = extractor.extract_from_file(test_file)
    
    # Verify dataframe
    assert not df.is_empty()
    assert len(df) == 3
    assert "id" in df.columns
    assert "start_date_local" in df.columns
    assert "file_source" in df.columns
    assert "file_hash" in df.columns
    assert "import_timestamp" in df.columns
    
    # Verify all records have the same file source
    assert df["file_source"].n_unique() == 1
    assert df["file_source"][0] == "test.csv"
    
    # Verify metadata
    assert metadata["file_name"] == "test.csv"
    assert metadata["file_path"] == str(test_file)
    assert metadata["file_hash"] is not None
    assert metadata["file_size"] > 0
    assert metadata["record_count"] == 3
    assert metadata["status"] == "success"


def test_extract_from_nonexistent_file(temp_bronze_dir):
    """Test handling of non-existent file."""
    extractor = CSVExtractor(bronze_path=str(temp_bronze_dir))
    
    fake_file = Path("/nonexistent/file.csv")
    df, metadata = extractor.extract_from_file(fake_file)
    
    assert df.is_empty()
    assert metadata["status"] == "error"
    assert "error" in metadata


def test_extract_from_corrupted_csv(temp_bronze_dir):
    """Test handling of corrupted CSV file."""
    extractor = CSVExtractor(bronze_path=str(temp_bronze_dir))
    
    # Create corrupted CSV
    bad_file = temp_bronze_dir / "incoming" / "bad.csv"
    bad_file.write_text("this,is,not,valid\n1,2")  # Wrong number of columns
    
    df, metadata = extractor.extract_from_file(bad_file)
    
    # Should handle gracefully
    assert metadata["file_path"] == str(bad_file)
    assert metadata["status"] in ["error", "success"]  # May parse partially


def test_get_incoming_files(temp_bronze_dir, sample_csv_content):
    """Test getting list of incoming files."""
    extractor = CSVExtractor(bronze_path=str(temp_bronze_dir))
    
    # Create test files
    for i in range(3):
        file_path = temp_bronze_dir / "incoming" / f"file{i}.csv"
        file_path.write_text(sample_csv_content)
    
    # Also create a non-CSV file that should be ignored
    (temp_bronze_dir / "incoming" / "readme.txt").write_text("ignore me")
    
    # Get incoming files
    files = extractor.get_incoming_files()
    
    assert len(files) == 3
    assert all(f.suffix == ".csv" for f in files)
    assert all(f.parent == temp_bronze_dir / "incoming" for f in files)


def test_get_archive_files(temp_bronze_dir, sample_csv_content):
    """Test getting list of archive files."""
    extractor = CSVExtractor(bronze_path=str(temp_bronze_dir))
    
    # Create archive files
    for i in range(2):
        file_path = temp_bronze_dir / "archive" / f"old_file{i}.csv"
        file_path.write_text(sample_csv_content)
    
    # Get archive files
    files = extractor.get_archive_files()
    
    assert len(files) == 2
    assert all(f.parent == temp_bronze_dir / "archive" for f in files)


def test_move_to_archive(temp_bronze_dir, sample_csv_content):
    """Test moving file to archive."""
    extractor = CSVExtractor(bronze_path=str(temp_bronze_dir))
    
    # Create test file
    incoming_file = temp_bronze_dir / "incoming" / "test.csv"
    incoming_file.write_text(sample_csv_content)
    
    # Move to archive
    archived_path = extractor.move_to_archive(incoming_file)
    
    # Verify move
    assert not incoming_file.exists()
    assert archived_path.exists()
    assert archived_path.parent == temp_bronze_dir / "archive"
    assert archived_path.name == "test.csv"
    
    # Verify content preserved
    assert archived_path.read_text() == sample_csv_content


def test_move_to_archive_with_existing_file(temp_bronze_dir, sample_csv_content):
    """Test moving file to archive when file already exists."""
    extractor = CSVExtractor(bronze_path=str(temp_bronze_dir))
    
    # Create existing archive file
    existing_archive = temp_bronze_dir / "archive" / "test.csv"
    existing_archive.write_text("old content")
    
    # Create new incoming file
    incoming_file = temp_bronze_dir / "incoming" / "test.csv"
    incoming_file.write_text(sample_csv_content)
    
    # Move to archive (should create timestamped version)
    archived_path = extractor.move_to_archive(incoming_file)
    
    # Verify move
    assert not incoming_file.exists()
    assert archived_path.exists()
    assert archived_path != existing_archive  # Should have different name
    assert "test" in archived_path.stem  # Should contain original name
    
    # Original archive file should still exist
    assert existing_archive.exists()
    assert existing_archive.read_text() == "old content"


def test_extract_all_with_multiple_files(temp_bronze_dir):
    """Test extracting from multiple CSV files."""
    extractor = CSVExtractor(bronze_path=str(temp_bronze_dir))
    
    # Create multiple CSV files with different data
    csv1 = """id,start_date_local,name,type,moving_time,distance
i1001,2024-01-01,Ride 1,Ride,3600,30000
i1002,2024-01-02,Ride 2,Ride,5400,45000"""
    
    csv2 = """id,start_date_local,name,type,moving_time,distance
i1003,2024-01-03,Ride 3,Ride,4500,35000
i1001,2024-01-01,Ride 1,Ride,3600,30000"""  # Duplicate
    
    (temp_bronze_dir / "incoming" / "file1.csv").write_text(csv1)
    (temp_bronze_dir / "incoming" / "file2.csv").write_text(csv2)
    
    # Extract all
    df, metadata = extractor.extract_all(include_archive=False)
    
    # Verify combined dataframe
    assert len(df) == 4  # 2 + 2 records (including duplicate)
    assert len(metadata) == 2  # 2 files
    
    # Verify file sources are tracked
    assert df["file_source"].n_unique() == 2
    assert "file1.csv" in df["file_source"].to_list()
    assert "file2.csv" in df["file_source"].to_list()
    
    # Verify metadata
    assert all(m["status"] == "success" for m in metadata)
    assert sum(m["record_count"] for m in metadata) == 4


def test_extract_all_with_archive(temp_bronze_dir):
    """Test extracting from both incoming and archive."""
    extractor = CSVExtractor(bronze_path=str(temp_bronze_dir))
    
    csv_content = """id,start_date_local,name,type,moving_time,distance
i1001,2024-01-01,Ride 1,Ride,3600,30000"""
    
    # Create files in both locations
    (temp_bronze_dir / "incoming" / "new.csv").write_text(csv_content)
    (temp_bronze_dir / "archive" / "old.csv").write_text(csv_content)
    
    # Extract without archive
    df1, metadata1 = extractor.extract_all(include_archive=False)
    assert len(df1) == 1
    assert len(metadata1) == 1
    
    # Extract with archive
    df2, metadata2 = extractor.extract_all(include_archive=True)
    assert len(df2) == 2
    assert len(metadata2) == 2


def test_data_integrity_preservation(temp_bronze_dir):
    """Test that exact data values are preserved during extraction."""
    extractor = CSVExtractor(bronze_path=str(temp_bronze_dir))
    
    # Create CSV with precise values
    csv_content = """id,start_date_local,name,icu_average_watts,distance,moving_time
i1234,2024-01-15 08:30:00,Test Ride,234.567,42195.5,3661.25"""
    
    test_file = temp_bronze_dir / "incoming" / "precision.csv"
    test_file.write_text(csv_content)
    
    # Extract
    df, metadata = extractor.extract_from_file(test_file)
    
    # Verify exact values are preserved
    assert df["id"][0] == "i1234"
    assert df["name"][0] == "Test Ride"
    assert df["icu_average_watts"][0] == 234.567
    assert df["distance"][0] == 42195.5
    assert df["moving_time"][0] == 3661.25
    
    # Verify hash integrity
    original_hash = hashlib.sha256(csv_content.encode()).hexdigest()
    file_hash = extractor.calculate_file_hash(test_file)
    assert file_hash == original_hash


def test_empty_csv_handling(temp_bronze_dir):
    """Test handling of empty CSV file."""
    extractor = CSVExtractor(bronze_path=str(temp_bronze_dir))
    
    # Create empty CSV (headers only)
    empty_csv = temp_bronze_dir / "incoming" / "empty.csv"
    empty_csv.write_text("id,start_date_local,name,type")
    
    df, metadata = extractor.extract_from_file(empty_csv)
    
    assert df.is_empty() or len(df) == 0
    assert metadata["record_count"] == 0
    assert metadata["status"] == "success"


def test_large_csv_performance(temp_bronze_dir):
    """Test performance with large CSV file."""
    extractor = CSVExtractor(bronze_path=str(temp_bronze_dir))
    
    # Generate large CSV
    rows = ["id,start_date_local,name,type,moving_time,distance"]
    for i in range(10000):
        rows.append(f"i{i},2024-01-01,Ride {i},Ride,3600,30000")
    
    large_csv = temp_bronze_dir / "incoming" / "large.csv"
    large_csv.write_text("\n".join(rows))
    
    # Time the extraction
    import time
    start = time.time()
    df, metadata = extractor.extract_from_file(large_csv)
    elapsed = time.time() - start
    
    # Verify extraction
    assert len(df) == 10000
    assert metadata["record_count"] == 10000
    
    # Performance check (should be fast with Polars)
    assert elapsed < 5.0  # Should process 10k records in under 5 seconds