"""Comprehensive tests for data validation and accuracy."""

import pytest
import polars as pl
import pandas as pd
from datetime import datetime
from typing import Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.validator import DataValidator, StrictActivityModel
from pydantic import ValidationError


class TestStrictActivityModel:
    """Test the Pydantic model validation."""
    
    def test_minimal_valid_activity(self):
        """Test that minimal required fields pass validation."""
        data = {
            'id': 'test_123',
            'start_date_local': datetime.now(),
            'name': 'Test Ride',
            'type': 'Ride'
        }
        
        activity = StrictActivityModel(**data)
        assert activity.id == 'test_123'
        assert activity.name == 'Test Ride'
        
    def test_missing_required_field_fails(self):
        """Test that missing required fields cause validation error."""
        data = {
            'start_date_local': datetime.now(),
            'name': 'Test Ride',
            'type': 'Ride'
        }
        
        with pytest.raises(ValidationError) as exc:
            StrictActivityModel(**data)
        assert 'id' in str(exc.value)
    
    def test_garmin_field_mapping(self):
        """Test that Garmin field names are properly handled."""
        data = {
            'id': 'garmin_123',
            'start_date_local': '2025-01-01T10:00:00',
            'name': 'Morning Ride',
            'type': 'road_biking',
            'moving_duration': 3600.0,  # Garmin uses moving_duration
            'elevation_gain': 100.0,  # Garmin uses elevation_gain
            'average_hr': 150,  # Garmin uses average_hr
            'average_power': 200.0,  # Garmin uses average_power
            'training_stress_score': 75.0  # Garmin TSS
        }
        
        activity = StrictActivityModel(**data)
        normalized = activity.normalize()
        
        # Check that normalization maps fields correctly
        assert normalized['moving_time'] == 3600.0
        assert normalized['total_elevation_gain'] == 100.0
        assert normalized['average_heartrate'] == 150
        assert normalized['icu_average_watts'] == 200.0
        assert normalized['icu_training_load'] == 75.0
        assert normalized['has_heartrate'] is True
        assert normalized['device_watts'] is True
    
    def test_intervals_icu_field_mapping(self):
        """Test that Intervals.icu field names are properly handled."""
        data = {
            'id': 'icu_123',
            'start_date_local': '2025-01-01T10:00:00',
            'name': 'Morning Ride',
            'type': 'Ride',
            'moving_time': 3600.0,
            'total_elevation_gain': 100.0,
            'average_heartrate': 150,
            'icu_average_watts': 200.0,
            'icu_training_load': 75.0
        }
        
        activity = StrictActivityModel(**data)
        normalized = activity.normalize()
        
        # Check that fields are preserved correctly
        assert normalized['moving_time'] == 3600.0
        assert normalized['total_elevation_gain'] == 100.0
        assert normalized['average_heartrate'] == 150
        assert normalized['icu_average_watts'] == 200.0
        assert normalized['icu_training_load'] == 75.0
    
    def test_empty_strings_converted_to_none(self):
        """Test that empty strings are converted to None."""
        data = {
            'id': 'test_123',
            'start_date_local': datetime.now(),
            'name': 'Test Ride',
            'type': 'Ride',
            'description': '   ',  # Empty/whitespace string
            'gear': ''  # Empty string
        }
        
        activity = StrictActivityModel(**data)
        assert activity.description is None
        assert activity.gear is None
    
    def test_json_field_handling(self):
        """Test that JSON fields are properly handled."""
        data = {
            'id': 'test_123',
            'start_date_local': datetime.now(),
            'name': 'Test Ride',
            'type': 'Ride',
            'hr_zones_data': [
                {'zone': 1, 'time': 600},
                {'zone': 2, 'time': 1200}
            ],
            'splits_data': {
                'laps': [{'distance': 1000, 'time': 300}]
            }
        }
        
        activity = StrictActivityModel(**data)
        # JSON fields should be converted to strings
        assert isinstance(activity.hr_zones_data, str)
        assert isinstance(activity.splits_data, str)


class TestDataValidator:
    """Test the DataValidator class."""
    
    def test_validate_single_activity(self):
        """Test validation of a single activity."""
        data = {
            'id': 'test_123',
            'start_date_local': '2025-01-01T10:00:00',
            'name': 'Test Ride',
            'type': 'Ride',
            'moving_duration': 3600.0,
            'average_power': 200.0
        }
        
        validated = DataValidator.validate_activity(data)
        
        assert validated['id'] == 'test_123'
        assert validated['moving_time'] == 3600.0
        assert validated['icu_average_watts'] == 200.0
        assert validated['device_watts'] is True
    
    def test_validate_dataframe_polars(self):
        """Test validation of a Polars DataFrame."""
        data = {
            'id': ['test_1', 'test_2'],
            'start_date_local': [datetime.now(), datetime.now()],
            'name': ['Ride 1', 'Ride 2'],
            'type': ['Ride', 'Ride'],
            'moving_duration': [3600.0, 4200.0],
            'average_hr': [150, 160]
        }
        
        df = pl.DataFrame(data)
        validated_df = DataValidator.validate_dataframe(df)
        
        assert len(validated_df) == 2
        assert 'moving_time' in validated_df.columns
        assert 'average_heartrate' in validated_df.columns
        assert validated_df['has_heartrate'].to_list() == [True, True]
    
    def test_validate_dataframe_pandas(self):
        """Test validation of a Pandas DataFrame."""
        data = {
            'id': ['test_1', 'test_2'],
            'start_date_local': [datetime.now(), datetime.now()],
            'name': ['Ride 1', 'Ride 2'],
            'type': ['Ride', 'Ride'],
            'moving_duration': [3600.0, 4200.0],
            'average_power': [200.0, 250.0]
        }
        
        df = pd.DataFrame(data)
        validated_df = DataValidator.validate_dataframe(df)
        
        assert len(validated_df) == 2
        assert 'moving_time' in validated_df.columns
        assert 'icu_average_watts' in validated_df.columns
        assert validated_df['device_watts'].to_list() == [True, True]
    
    def test_validation_error_handling(self):
        """Test that validation errors are properly handled."""
        data = {
            'id': ['test_1', None],  # Second row missing ID
            'start_date_local': [datetime.now(), datetime.now()],
            'name': ['Ride 1', 'Ride 2'],
            'type': ['Ride', 'Ride']
        }
        
        df = pl.DataFrame(data)
        
        # Should still return valid rows
        validated_df = DataValidator.validate_dataframe(df)
        assert len(validated_df) == 1  # Only first row is valid
        assert validated_df['id'][0] == 'test_1'
    
    def test_data_quality_metrics(self):
        """Test data quality assessment."""
        data = {
            'id': ['test_1', 'test_2', 'test_3'],
            'start_date_local': [datetime.now(), datetime.now(), datetime.now()],
            'name': ['Ride 1', 'Ride 2', 'Ride 3'],
            'type': ['Ride', 'Ride', 'Ride'],
            'moving_time': [3600.0, None, 4200.0],
            'distance': [30000.0, 25000.0, 35000.0],
            'average_heartrate': [150, 160, None]
        }
        
        df = pl.DataFrame(data)
        metrics = DataValidator.ensure_data_quality(df)
        
        assert metrics['total_rows'] == 3
        assert 'moving_time' in metrics['null_percentages']
        assert metrics['null_percentages']['moving_time'] == pytest.approx(33.33, 0.1)
        assert metrics['quality_score'] > 0
        assert 'distance' in metrics['value_ranges']
        assert metrics['value_ranges']['distance']['min'] == 25000.0
        assert metrics['value_ranges']['distance']['max'] == 35000.0


class TestDataConsistency:
    """Test data consistency across different sources."""
    
    def test_garmin_intervals_compatibility(self):
        """Test that Garmin and Intervals.icu data can coexist."""
        garmin_data = {
            'id': 'garmin_123',
            'start_date_local': '2025-01-01T10:00:00',
            'name': 'Garmin Ride',
            'type': 'road_biking',
            'moving_duration': 3600.0,
            'average_power': 200.0,
            'training_stress_score': 75.0
        }
        
        intervals_data = {
            'id': 'icu_456',
            'start_date_local': '2025-01-02T10:00:00',
            'name': 'Intervals Ride',
            'type': 'Ride',
            'moving_time': 3700.0,
            'icu_average_watts': 210.0,
            'icu_training_load': 80.0
        }
        
        # Validate both
        garmin_validated = DataValidator.validate_activity(garmin_data)
        intervals_validated = DataValidator.validate_activity(intervals_data)
        
        # Both should have consistent field names after normalization
        assert 'moving_time' in garmin_validated
        assert 'moving_time' in intervals_validated
        assert 'icu_average_watts' in garmin_validated
        assert 'icu_average_watts' in intervals_validated
        assert 'icu_training_load' in garmin_validated
        assert 'icu_training_load' in intervals_validated
    
    def test_mixed_source_dataframe(self):
        """Test validation of dataframe with mixed sources."""
        data = {
            'id': ['garmin_1', 'icu_1', 'garmin_2'],
            'start_date_local': [datetime.now()] * 3,
            'name': ['Ride 1', 'Ride 2', 'Ride 3'],
            'type': ['road_biking', 'Ride', 'road_biking'],
            # Mix of field names
            'moving_duration': [3600.0, None, 4200.0],
            'moving_time': [None, 3700.0, None],
            'average_power': [200.0, None, 250.0],
            'icu_average_watts': [None, 210.0, None]
        }
        
        df = pl.DataFrame(data)
        validated_df = DataValidator.validate_dataframe(df)
        
        # All should have normalized fields
        assert len(validated_df) == 3
        assert validated_df['moving_time'].is_not_null().all()
        assert validated_df['icu_average_watts'].is_not_null().all()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])