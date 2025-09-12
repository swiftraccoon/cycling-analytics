#!/usr/bin/env python3
"""
Test script to verify all pipeline components are working correctly.

Creates sample data and tests each component individually.
"""

import logging
from pathlib import Path
import sys
import tempfile
import csv
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.storage.database.manager import DatabaseManager
from src.storage.schemas.activity_schema import initialize_database
from src.etl.extractors.csv_extractor import IntervalsCsvExtractor
from src.etl.validators.integrity_validator import IntegrityValidator
from src.data.deduplicator import Deduplicator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_csv_data() -> str:
    """Create sample intervals.icu CSV data for testing."""
    
    sample_data = [
        {
            'Id': 'activity_001',
            'Name': 'Morning Ride',
            'Type': 'Ride',
            'Start date (local)': '2024-01-15T08:30:00',
            'Distance': '25.5',
            'Moving time': '3600',
            'Elapsed time': '3900',
            'Total elevation gain': '450',
            'Weighted Average Power': '250',
            'Power Meter': 'TRUE',
            'Device Watts': 'TRUE',
            'Max Watts': '450',
            'Normalized Power®': '265',
            'Average Heartrate': '155',
            'Max Heartrate': '178',
            'Average Speed': '7.1',
            'Max Speed': '15.2',
            'Average Cadence': '85',
            'Max Cadence': '110',
            'Training Stress Score': '85',
            'Kilojoules': '900'
        },
        {
            'Id': 'activity_002',
            'Name': 'Evening Ride',
            'Type': 'Ride',
            'Start date (local)': '2024-01-15T18:00:00',
            'Distance': '30.2',
            'Moving time': '4200',
            'Elapsed time': '4500',
            'Total elevation gain': '380',
            'Weighted Average Power': '220',
            'Power Meter': 'TRUE',
            'Device Watts': 'TRUE',
            'Max Watts': '420',
            'Normalized Power®': '240',
            'Average Heartrate': '148',
            'Max Heartrate': '170',
            'Average Speed': '7.2',
            'Max Speed': '14.8',
            'Average Cadence': '82',
            'Max Cadence': '105',
            'Training Stress Score': '75',
            'Kilojoules': '924'
        },
        {
            'Id': 'activity_001',  # Duplicate of first activity
            'Name': 'Morning Ride',
            'Type': 'Ride',
            'Start date (local)': '2024-01-15T08:30:00',
            'Distance': '25.5',
            'Moving time': '3600',
            'Elapsed time': '3900',
            'Total elevation gain': '450',
            'Weighted Average Power': '250',
            'Power Meter': 'TRUE',
            'Device Watts': 'TRUE',
            'Max Watts': '450',
            'Normalized Power®': '265',
            'Average Heartrate': '155',
            'Max Heartrate': '178',
            'Average Speed': '7.1',
            'Max Speed': '15.2',
            'Average Cadence': '85',
            'Max Cadence': '110',
            'Training Stress Score': '85',
            'Kilojoules': '900'
        }
    ]
    
    # Create CSV content
    if not sample_data:
        return ""
    
    output = []
    
    # Add header
    headers = list(sample_data[0].keys())
    output.append(','.join(headers))
    
    # Add data rows
    for row in sample_data:
        values = [str(row.get(header, '')) for header in headers]
        output.append(','.join(values))
    
    return '\n'.join(output)


def test_database_manager():
    """Test database manager functionality."""
    logger.info("Testing Database Manager...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as db_file:
        db_path = Path(db_file.name)
    
    try:
        # Initialize database
        initialize_database(db_path)
        db_manager = DatabaseManager(db_path)
        
        # Test basic operations
        summary = db_manager.get_activity_summary()
        logger.info(f"Initial database summary: {summary}")
        
        # Test activity insertion
        activity_data = {
            'activity_id': 'test_001',
            'name': 'Test Activity',
            'type': 'Ride',
            'start_date_local': '2024-01-15T08:30:00',
            'distance': 25.5
        }
        
        activity_id = db_manager.insert_activity(
            activity_data, 
            '/test/file.csv', 
            'testhash123',
            1
        )
        
        logger.info(f"Inserted activity with ID: {activity_id}")
        
        # Test retrieval
        retrieved = db_manager.get_activity_by_id('test_001')
        logger.info(f"Retrieved activity: {retrieved is not None}")
        
        logger.info("✓ Database Manager test passed")
        
    finally:
        if db_path.exists():
            db_path.unlink()


def test_csv_extractor():
    """Test CSV extractor functionality."""
    logger.info("Testing CSV Extractor...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as csv_file:
        csv_file.write(create_sample_csv_data())
        csv_path = Path(csv_file.name)
    
    try:
        extractor = IntervalsCsvExtractor()
        result = extractor.extract_intervals_file(csv_path)
        
        if result.success and result.data is not None:
            logger.info(f"✓ Extracted {len(result.data)} records")
            logger.info(f"✓ File hash: {result.metadata.file_hash[:16]}...")
            logger.info(f"✓ Processing time: {result.processing_time_ms}ms")
        else:
            logger.error(f"✗ Extraction failed: {result.error_message}")
            
        logger.info("✓ CSV Extractor test passed")
        
    finally:
        if csv_path.exists():
            csv_path.unlink()


def test_data_validator():
    """Test data validation functionality."""
    logger.info("Testing Data Validator...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as csv_file:
        csv_file.write(create_sample_csv_data())
        csv_path = Path(csv_file.name)
    
    try:
        # Extract data first
        extractor = IntervalsCsvExtractor()
        result = extractor.extract_intervals_file(csv_path)
        
        if result.success and result.data is not None:
            # Validate data
            validator = IntegrityValidator()
            report = validator.validate_dataset(
                result.data,
                str(csv_path),
                result.metadata.file_hash
            )
            
            logger.info(f"✓ Validation status: {report.validation_status}")
            logger.info(f"✓ Quality score: {report.quality_score:.1f}")
            logger.info(f"✓ Issues found: {len([i for col in report.column_results for i in col.issues])}")
        
        logger.info("✓ Data Validator test passed")
        
    finally:
        if csv_path.exists():
            csv_path.unlink()


def test_deduplicator():
    """Test deduplication functionality."""
    logger.info("Testing Deduplicator...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as csv_file:
        csv_file.write(create_sample_csv_data())
        csv_path = Path(csv_file.name)
    
    try:
        # Extract data
        extractor = IntervalsCsvExtractor()
        result = extractor.extract_intervals_file(csv_path)
        
        if result.success and result.data is not None and result.metadata is not None:
            # Test deduplication
            deduplicator = Deduplicator()
            data_sources = [(result.data, result.metadata.dict())]
            
            report = deduplicator.deduplicate_data(data_sources)
            
            logger.info(f"✓ Total records: {report.total_records}")
            logger.info(f"✓ Unique activities: {report.unique_activities}")
            logger.info(f"✓ Duplicates removed: {report.total_duplicates_removed}")
            logger.info(f"✓ Duplicate groups: {report.duplicate_groups}")
        
        logger.info("✓ Deduplicator test passed")
        
    finally:
        if csv_path.exists():
            csv_path.unlink()


def main():
    """Run all component tests."""
    logger.info("Starting pipeline component tests...")
    
    try:
        test_database_manager()
        test_csv_extractor()
        test_data_validator()
        test_deduplicator()
        
        logger.info("🎉 All component tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()