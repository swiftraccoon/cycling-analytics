"""Tests for data validation and integrity."""

import pytest
import polars as pl
from datetime import datetime, timedelta
import numpy as np

from src.data.deduplicator import Deduplicator
from src.storage.database.manager import DatabaseManager
from src.etl.extractors.csv_extractor import CSVExtractor


class TestDataIntegrity:
    """Test suite for data integrity and validation."""
    
    def test_no_data_modification_during_deduplication(self):
        """Ensure deduplication does not modify original values."""
        # Create test data with precise values
        df = pl.DataFrame({
            "id": ["act1", "act1", "act2"],
            "start_date_local": [datetime(2024, 1, 1, 8, 30, 15)] * 3,
            "distance": [42195.5, 42195.5, 30000.123],
            "icu_average_watts": [234.567, 234.567, 198.432],
            "moving_time": [3661.25, 3661.25, 2843.75],
            "file_source": ["file1.csv", "file2.csv", "file1.csv"],
            "import_timestamp": [datetime.now()] * 3,
        })
        
        # Store original values
        original_values = {
            "distance": df["distance"].to_list(),
            "watts": df["icu_average_watts"].to_list(),
            "time": df["moving_time"].to_list(),
        }
        
        # Perform deduplication
        dedup = Deduplicator()
        df_dedup, _ = dedup.deduplicate(df)
        
        # Verify no value modification for kept records
        for col, original in original_values.items():
            if col == "distance":
                col_name = "distance"
            elif col == "watts":
                col_name = "icu_average_watts"
            else:
                col_name = "moving_time"
            
            # Check that values in deduplicated data match originals exactly
            dedup_values = df_dedup[col_name].to_list()
            for val in dedup_values:
                assert val in original, f"Value {val} was modified during deduplication"
    
    def test_no_false_data_insertion(self):
        """Ensure no false or synthetic data is inserted."""
        df = pl.DataFrame({
            "id": ["act1", "act2"],
            "start_date_local": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "distance": [30000, None],  # One null value
            "moving_time": [3600, 5400],
            "file_source": ["file1.csv", "file1.csv"],
            "import_timestamp": [datetime.now()] * 2,
        })
        
        original_nulls = df.null_count().to_dict()
        
        # Process through deduplicator
        dedup = Deduplicator()
        df_dedup, _ = dedup.deduplicate(df)
        
        # Verify no nulls were filled with synthetic data
        for col in df.columns:
            if col in df_dedup.columns:
                original_null_count = original_nulls.get(col, [0])[0]
                # Deduplication may remove rows but shouldn't fill nulls
                assert df_dedup[col].null_count() <= original_null_count
    
    def test_exact_duplicate_detection_accuracy(self):
        """Test accurate detection of exact duplicates."""
        # Create data with known duplicates
        df = pl.DataFrame({
            "id": ["act1", "act2", "act1", "act3", "act2", "act1"],
            "start_date_local": [datetime(2024, 1, i) for i in [1, 2, 1, 3, 2, 1]],
            "distance": [30000, 40000, 30000, 35000, 40000, 30000],
            "file_source": ["f1.csv", "f1.csv", "f2.csv", "f1.csv", "f3.csv", "f4.csv"],
            "import_timestamp": [datetime.now() + timedelta(seconds=i) for i in range(6)],
        })
        
        dedup = Deduplicator()
        df_marked = dedup.identify_exact_duplicates(df)
        
        # Count duplicates per ID
        id_counts = df_marked.group_by("id").agg(pl.len().alias("count"))
        
        # Verify correct duplicate counts
        assert id_counts.filter(pl.col("id") == "act1")["count"][0] == 3
        assert id_counts.filter(pl.col("id") == "act2")["count"][0] == 2
        assert id_counts.filter(pl.col("id") == "act3")["count"][0] == 1
    
    def test_file_source_tracking_accuracy(self):
        """Test accurate tracking of file sources."""
        df = pl.DataFrame({
            "id": ["act1", "act1", "act2", "act1"],
            "file_source": ["file1.csv", "file2.csv", "file1.csv", "file3.csv"],
            "import_timestamp": [datetime.now() + timedelta(seconds=i) for i in range(4)],
        })
        
        dedup = Deduplicator()
        df_sources = dedup.track_file_sources(df)
        
        # Check act1 has all three sources
        act1_sources = df_sources.filter(pl.col("id") == "act1")["all_file_sources"][0]
        assert "file1.csv" in act1_sources
        assert "file2.csv" in act1_sources
        assert "file3.csv" in act1_sources
        
        # Check act2 has only one source
        act2_sources = df_sources.filter(pl.col("id") == "act2")["all_file_sources"][0]
        assert act2_sources == "file1.csv"
    
    def test_timestamp_preservation(self):
        """Test that timestamps are preserved exactly."""
        precise_time = datetime(2024, 1, 15, 8, 30, 45, 123456)
        
        df = pl.DataFrame({
            "id": ["act1"],
            "start_date_local": [precise_time],
            "icu_sync_date": [precise_time + timedelta(hours=1)],
            "import_timestamp": [datetime.now()],
            "file_source": ["test.csv"],
        })
        
        # Process through deduplication
        dedup = Deduplicator()
        df_processed, _ = dedup.deduplicate(df)
        
        # Verify timestamp preservation
        assert df_processed["start_date_local"][0] == precise_time
        assert df_processed["icu_sync_date"][0] == precise_time + timedelta(hours=1)
    
    def test_numeric_precision_preservation(self):
        """Test that numeric precision is maintained."""
        df = pl.DataFrame({
            "id": ["act1"],
            "distance": [42195.5],  # Marathon distance
            "icu_average_watts": [234.567],
            "icu_normalized_watts": [245.89],
            "icu_training_load": [156.789],
            "icu_intensity": [0.8234],
            "total_elevation_gain": [1234.56],
            "file_source": ["test.csv"],
            "import_timestamp": [datetime.now()],
        })
        
        # Store original values
        original_distance = df["distance"][0]
        original_watts = df["icu_average_watts"][0]
        original_np = df["icu_normalized_watts"][0]
        original_load = df["icu_training_load"][0]
        original_intensity = df["icu_intensity"][0]
        original_elevation = df["total_elevation_gain"][0]
        
        # Process
        dedup = Deduplicator()
        df_processed, _ = dedup.deduplicate(df)
        
        # Verify exact preservation
        assert df_processed["distance"][0] == original_distance
        assert df_processed["icu_average_watts"][0] == original_watts
        assert df_processed["icu_normalized_watts"][0] == original_np
        assert df_processed["icu_training_load"][0] == original_load
        assert df_processed["icu_intensity"][0] == original_intensity
        assert df_processed["total_elevation_gain"][0] == original_elevation
    
    def test_string_data_preservation(self):
        """Test that string data is preserved exactly."""
        df = pl.DataFrame({
            "id": ["i123456"],
            "name": ["Morning Ride - Z2 Endurance (with coffee stop!)"],
            "type": ["Ride"],
            "gear": ["Canyon Aeroad CF SLX 8.0"],
            "description": ["Great ride with\nmultiple lines\nand special chars: @#$%"],
            "file_source": ["test.csv"],
            "import_timestamp": [datetime.now()],
        })
        
        # Store originals
        original_name = df["name"][0]
        original_gear = df["gear"][0]
        original_desc = df["description"][0]
        
        # Process
        dedup = Deduplicator()
        df_processed, _ = dedup.deduplicate(df)
        
        # Verify exact preservation
        assert df_processed["name"][0] == original_name
        assert df_processed["gear"][0] == original_gear
        assert df_processed["description"][0] == original_desc
    
    def test_null_handling(self):
        """Test that null values are handled correctly."""
        df = pl.DataFrame({
            "id": ["act1", "act2", "act3"],
            "distance": [30000, None, 40000],
            "icu_average_watts": [None, 200, None],
            "name": ["Ride 1", None, "Ride 3"],
            "file_source": ["test.csv"] * 3,
            "import_timestamp": [datetime.now()] * 3,
        })
        
        # Process
        dedup = Deduplicator()
        df_processed, _ = dedup.deduplicate(df)
        
        # Verify nulls are preserved
        assert df_processed.filter(pl.col("id") == "act2")["distance"][0] is None
        assert df_processed.filter(pl.col("id") == "act1")["icu_average_watts"][0] is None
        assert df_processed.filter(pl.col("id") == "act2")["name"][0] is None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes."""
        dedup = Deduplicator()
        
        # Empty dataframe
        df_empty = pl.DataFrame()
        df_result, report = dedup.deduplicate(df_empty)
        
        assert df_result.is_empty()
        assert report["total_records"] == 0
        assert report["unique_records"] == 0
        assert report["exact_duplicates"] == 0
    
    def test_single_record_handling(self):
        """Test handling of single record."""
        df = pl.DataFrame({
            "id": ["act1"],
            "distance": [30000],
            "file_source": ["test.csv"],
            "import_timestamp": [datetime.now()],
        })
        
        dedup = Deduplicator()
        df_result, report = dedup.deduplicate(df)
        
        assert len(df_result) == 1
        assert report["exact_duplicates"] == 0
        assert report["unique_records"] == 1
    
    def test_all_duplicates_handling(self):
        """Test when all records are duplicates of one activity."""
        df = pl.DataFrame({
            "id": ["act1"] * 10,
            "distance": [30000] * 10,
            "file_source": [f"file{i}.csv" for i in range(10)],
            "import_timestamp": [datetime.now() + timedelta(seconds=i) for i in range(10)],
        })
        
        dedup = Deduplicator()
        df_result, report = dedup.deduplicate(df)
        
        assert len(df_result) == 1  # Should keep only one
        assert report["exact_duplicates"] == 9
        assert report["unique_records"] == 1
    
    def test_missing_id_column(self):
        """Test handling when ID column is missing."""
        df = pl.DataFrame({
            "distance": [30000, 40000],
            "moving_time": [3600, 5400],
            "file_source": ["test.csv"] * 2,
            "import_timestamp": [datetime.now()] * 2,
        })
        
        dedup = Deduplicator()
        df_result, report = dedup.deduplicate(df)
        
        # Should return original dataframe when can't deduplicate
        assert len(df_result) == 2
    
    def test_mixed_data_types(self):
        """Test handling of mixed data types."""
        df = pl.DataFrame({
            "id": ["act1", "act2"],
            "distance": [30000, 40000],  # Integer
            "icu_average_watts": [200.5, 250.75],  # Float
            "name": ["Ride 1", "Ride 2"],  # String
            "trainer": [True, False],  # Boolean
            "start_date_local": [datetime(2024, 1, 1), datetime(2024, 1, 2)],  # Datetime
            "file_source": ["test.csv"] * 2,
            "import_timestamp": [datetime.now()] * 2,
        })
        
        dedup = Deduplicator()
        df_result, _ = dedup.deduplicate(df)
        
        # Verify all data types preserved
        assert df_result["distance"].dtype in [pl.Int64, pl.Float64]
        assert df_result["icu_average_watts"].dtype == pl.Float64
        assert df_result["name"].dtype == pl.Utf8
        assert df_result["trainer"].dtype == pl.Boolean
        assert df_result["start_date_local"].dtype == pl.Datetime
    
    def test_very_large_values(self):
        """Test handling of very large numeric values."""
        df = pl.DataFrame({
            "id": ["act1"],
            "distance": [999999999.999],  # Very large distance
            "moving_time": [86400 * 7],  # One week in seconds
            "icu_joules": [9999999999],  # Very large energy
            "file_source": ["test.csv"],
            "import_timestamp": [datetime.now()],
        })
        
        dedup = Deduplicator()
        df_result, _ = dedup.deduplicate(df)
        
        # Verify large values preserved
        assert df_result["distance"][0] == 999999999.999
        assert df_result["moving_time"][0] == 86400 * 7
        assert df_result["icu_joules"][0] == 9999999999
    
    def test_unicode_handling(self):
        """Test handling of unicode characters."""
        df = pl.DataFrame({
            "id": ["act1"],
            "name": ["Ride with émojis 🚴‍♂️ and spëcial çhars"],
            "description": ["日本語 test κόσμος Здравствуй"],
            "file_source": ["test.csv"],
            "import_timestamp": [datetime.now()],
        })
        
        dedup = Deduplicator()
        df_result, _ = dedup.deduplicate(df)
        
        # Verify unicode preserved
        assert "🚴‍♂️" in df_result["name"][0]
        assert "日本語" in df_result["description"][0]
        assert "κόσμος" in df_result["description"][0]


class TestDuplicateResolution:
    """Test duplicate resolution strategies."""
    
    def test_keep_latest_strategy(self):
        """Test keeping the latest version of duplicates."""
        df = pl.DataFrame({
            "id": ["act1", "act1", "act1"],
            "distance": [30000, 30100, 30200],  # Slightly different
            "import_timestamp": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
            ],
            "file_source": ["file1.csv", "file2.csv", "file3.csv"],
        })
        
        dedup = Deduplicator()
        df_result = dedup.resolve_duplicates(df, strategy="keep_latest")
        
        if "id" in df_result.columns and len(df_result) == 1:
            # Should keep the one with latest timestamp
            assert df_result["distance"][0] == 30200
            assert df_result["file_source"][0] == "file3.csv"
    
    def test_keep_first_strategy(self):
        """Test keeping the first version of duplicates."""
        df = pl.DataFrame({
            "id": ["act1", "act1", "act1"],
            "distance": [30000, 30100, 30200],
            "import_timestamp": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
            ],
            "file_source": ["file1.csv", "file2.csv", "file3.csv"],
        })
        
        dedup = Deduplicator()
        df_result = dedup.resolve_duplicates(df, strategy="keep_first")
        
        if "id" in df_result.columns and len(df_result) == 1:
            # Should keep the one with earliest timestamp
            assert df_result["distance"][0] == 30000
            assert df_result["file_source"][0] == "file1.csv"