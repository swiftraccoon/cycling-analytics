"""Integration tests for the cycling analytics platform."""

import pytest
import polars as pl
from pathlib import Path
import tempfile

from src.etl.extractors.csv_extractor import CSVExtractor
from src.data.deduplicator import Deduplicator
from src.storage.database.manager import DatabaseManager
from src.analytics.performance import PerformanceAnalyzer
from src.ml.predictions import PerformancePredictor
from src.export.reports import ReportExporter


def test_full_pipeline_integration(test_data_dir, bronze_layer, create_test_csv):
    """Test the complete data pipeline from CSV to analytics."""
    
    # Step 1: Create test CSV files
    csv_file1 = create_test_csv("activities_jan.csv")
    csv_file2 = create_test_csv("activities_feb.csv")
    
    # Step 2: Extract data
    extractor = CSVExtractor(bronze_path=str(bronze_layer))
    df, metadata = extractor.extract_all(include_archive=False)
    
    assert not df.is_empty()
    assert len(metadata) == 2
    
    # Step 3: Deduplicate
    deduplicator = Deduplicator()
    df_dedup, dedup_report = deduplicator.deduplicate(df)
    
    assert len(df_dedup) <= len(df)
    assert dedup_report["total_records"] == len(df)
    
    # Step 4: Save to database
    db_path = test_data_dir / "test.db"
    db = DatabaseManager(db_path=str(db_path))
    save_stats = db.save_activities(df_dedup)
    
    assert save_stats["saved"] > 0
    
    # Step 5: Retrieve and analyze
    activities = db.get_activities()
    assert not activities.is_empty()
    
    analyzer = PerformanceAnalyzer(activities)
    
    # Test FTP progression
    ftp_data = analyzer.calculate_ftp_progression()
    assert not ftp_data.is_empty()
    
    # Test training load
    training_load = analyzer.calculate_training_load(days=30)
    assert "activities" in training_load
    assert training_load["activities"] > 0
    
    # Test weekly summary
    weekly = analyzer.calculate_weekly_summary(weeks=4)
    assert not weekly.is_empty()
    
    # Step 6: Test predictions
    predictor = PerformancePredictor(activities)
    
    ftp_pred = predictor.predict_ftp_progression(days_ahead=30)
    # May have insufficient data for prediction
    if "error" not in ftp_pred:
        assert "current_ftp" in ftp_pred
        assert "predictions" in ftp_pred
    
    readiness = predictor.predict_performance_readiness()
    if "error" not in readiness:
        assert "overall_readiness" in readiness
    
    # Step 7: Test export
    exporter = ReportExporter(activities)
    
    # Test Excel export
    excel_path = test_data_dir / "test_report.xlsx"
    exported_path = exporter.export_to_excel(str(excel_path))
    assert Path(exported_path).exists()
    
    # Test CSV export
    csv_dir = test_data_dir / "csv_export"
    csv_files = exporter.export_to_csv(str(csv_dir))
    assert len(csv_files) > 0
    assert all(Path(f).exists() for f in csv_files.values())
    
    # Test summary report
    summary = exporter.generate_summary_report()
    assert "CYCLING ANALYTICS SUMMARY REPORT" in summary
    assert "Total Activities:" in summary


def test_continuous_update_workflow(test_data_dir, bronze_layer, sample_csv_data):
    """Test continuous update workflow with overlapping data."""
    
    # Initial data load
    extractor = CSVExtractor(bronze_path=str(bronze_layer))
    db_path = test_data_dir / "test.db"
    db = DatabaseManager(db_path=str(db_path))
    
    # First batch of activities
    batch1 = sample_csv_data.head(10)
    csv_file1 = bronze_layer / "incoming" / "batch1.csv"
    batch1.write_csv(csv_file1)
    
    df1, _ = extractor.extract_all()
    dedup = Deduplicator()
    df1_dedup, _ = dedup.deduplicate(df1)
    stats1 = db.save_activities(df1_dedup)
    
    assert stats1["saved"] == 10
    
    # Move to archive
    extractor.move_to_archive(csv_file1)
    assert not csv_file1.exists()
    assert (bronze_layer / "archive" / "batch1.csv").exists()
    
    # Second batch with some overlapping activities
    batch2 = sample_csv_data.slice(5, 10)  # Activities 5-14 (5 overlap)
    csv_file2 = bronze_layer / "incoming" / "batch2.csv"
    batch2.write_csv(csv_file2)
    
    df2, _ = extractor.extract_all(include_archive=True)  # Include archive
    df2_dedup, report2 = dedup.deduplicate(df2)
    stats2 = db.save_activities(df2_dedup)
    
    # Should detect the 5 overlapping activities
    assert report2["exact_duplicates"] == 5
    assert stats2["saved"] == 5  # Only new activities
    
    # Verify total in database
    all_activities = db.get_activities()
    assert len(all_activities) == 15  # 10 + 5 new


def test_data_integrity_preservation(bronze_layer, create_test_csv):
    """Test that data integrity is preserved throughout the pipeline."""
    
    # Create CSV with specific values
    test_data = pl.DataFrame({
        "id": ["test123"],
        "start_date_local": ["2024-01-15"],
        "name": ["Test Ride"],
        "icu_average_watts": [234.567],
        "icu_normalized_watts": [245.123],
        "icu_training_load": [67.89],
        "distance": [42195.5],  # Marathon distance
    })
    
    csv_file = bronze_layer / "incoming" / "integrity_test.csv"
    test_data.write_csv(csv_file)
    
    # Extract
    extractor = CSVExtractor(bronze_path=str(bronze_layer))
    df, metadata = extractor.extract_from_file(csv_file)
    
    # Verify hash was calculated
    assert metadata["file_hash"] is not None
    assert len(metadata["file_hash"]) == 64  # SHA-256 hash
    
    # Verify data values are preserved
    assert df["id"][0] == "test123"
    assert df["name"][0] == "Test Ride"
    assert abs(df["icu_average_watts"][0] - 234.567) < 0.001
    assert abs(df["distance"][0] - 42195.5) < 0.1
    
    # Process through deduplication
    dedup = Deduplicator()
    df_dedup, _ = dedup.deduplicate(df)
    
    # Values should still be unchanged
    assert df_dedup["icu_average_watts"][0] == df["icu_average_watts"][0]
    assert df_dedup["distance"][0] == df["distance"][0]


def test_error_handling(test_data_dir, bronze_layer):
    """Test error handling throughout the pipeline."""
    
    # Test with non-existent file
    extractor = CSVExtractor(bronze_path=str(bronze_layer))
    df, metadata = extractor.extract_from_file(Path("nonexistent.csv"))
    assert df.is_empty()
    assert metadata["status"] == "error"
    
    # Test with corrupted CSV
    bad_csv = bronze_layer / "incoming" / "bad.csv"
    bad_csv.write_text("this,is,not,valid,csv\n1,2")  # Wrong number of columns
    
    df, metadata = extractor.extract_from_file(bad_csv)
    # Should handle gracefully
    assert metadata["file_path"] == str(bad_csv)
    
    # Test database with invalid data
    db = DatabaseManager(db_path=str(test_data_dir / "test.db"))
    invalid_df = pl.DataFrame({"invalid_column": [1, 2, 3]})
    
    stats = db.save_activities(invalid_df)
    assert stats["saved"] == 0  # Should not save without required columns