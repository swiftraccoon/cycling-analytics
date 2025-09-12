"""Comprehensive tests for FIT data validation with Pydantic models.

NO COMPROMISES ON DATA QUALITY - 95% MINIMUM!
"""

import unittest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.fit_data import (
    FITSessionData, FITLapData, FITRecordData, 
    FITDeviceInfo, FITParseResult, FITSummaryStats
)
from src.integrations.fit_parser import FITParser
from src.data.validator import StrictActivityModel, DataValidator
from src.storage.database.manager import DatabaseManager
import polars as pl


class TestFITDataValidation(unittest.TestCase):
    """Test FIT data validation with strict Pydantic models."""
    
    def setUp(self):
        """Set up test environment."""
        self.parser = FITParser()
        self.maxDiff = None  # Show full diffs on failures
        
    def test_fit_session_data_validation(self):
        """Test that FIT session data passes strict Pydantic validation."""
        session_data = {
            'timestamp': datetime.now(timezone.utc),
            'start_time': datetime.now(timezone.utc),
            'total_elapsed_time': 3600.5,
            'total_timer_time': 3550.0,
            'total_distance': 25000.0,
            'avg_power': 200,
            'max_power': 500,
            'normalized_power': 220,
            'threshold_power': 250,  # CRITICAL FIELD
            'avg_cadence': 85,
            'max_cadence': 110,
            'avg_temperature': 22,
            'max_temperature': 28,
            'left_right_balance': 49.5,
            'avg_left_torque_effectiveness': 78.5,
            'avg_right_torque_effectiveness': 79.2,
            'avg_left_pedal_smoothness': 35.5,
            'avg_right_pedal_smoothness': 36.2,
            'training_stress_score': 95.5,
            'intensity_factor': 0.88,
            'total_work': 720000,
            'total_calories': 650
        }
        
        # Validate with Pydantic
        validated = FITSessionData(**session_data)
        
        # Verify critical fields are preserved
        self.assertEqual(validated.threshold_power, 250)
        self.assertEqual(validated.avg_cadence, 85)
        self.assertEqual(validated.avg_temperature, 22)
        self.assertEqual(validated.left_right_balance, 49.5)
        self.assertIsNotNone(validated.training_stress_score)
        
    def test_fit_device_info_validation(self):
        """Test device info validation."""
        device_data = {
            'manufacturer': 'Garmin',
            'product': 'Edge 830',
            'serial_number': 123456789,
            'software_version': 9.10,
            'product_name': 'Edge 830'
        }
        
        validated = FITDeviceInfo(**device_data)
        
        self.assertEqual(validated.manufacturer, 'Garmin')
        self.assertEqual(validated.product, 'Edge 830')
        self.assertEqual(validated.serial_number, '123456789')  # Converted to string
        
    def test_fit_lap_data_validation(self):
        """Test lap data validation."""
        lap_data = {
            'timestamp': datetime.now(timezone.utc),
            'start_time': datetime.now(timezone.utc),
            'total_elapsed_time': 600.0,
            'total_timer_time': 595.0,
            'total_distance': 5000.0,
            'avg_power': 210,
            'max_power': 450,
            'normalized_power': 225,
            'avg_cadence': 87,
            'max_cadence': 105,
            'avg_heart_rate': 155,
            'max_heart_rate': 172
        }
        
        validated = FITLapData(**lap_data)
        
        self.assertEqual(validated.avg_power, 210)
        self.assertEqual(validated.avg_cadence, 87)
        self.assertIsNotNone(validated.normalized_power)
        
    def test_fit_record_data_validation(self):
        """Test individual record validation."""
        record_data = {
            'timestamp': datetime.now(timezone.utc),
            'position_lat': 35.6762,
            'position_long': 139.6503,
            'distance': 1000.0,
            'speed': 8.5,
            'altitude': 150.0,
            'heart_rate': 145,
            'cadence': 85,
            'power': 200,
            'temperature': 22,
            'left_right_balance': 50.0,
            'left_torque_effectiveness': 80.0,
            'right_torque_effectiveness': 79.0
        }
        
        validated = FITRecordData(**record_data)
        
        self.assertEqual(validated.power, 200)
        self.assertEqual(validated.cadence, 85)
        self.assertEqual(validated.temperature, 22)
        self.assertEqual(validated.left_right_balance, 50.0)
        
    def test_fit_parse_result_validation(self):
        """Test complete FIT parse result validation."""
        parse_result = {
            'file_path': '/path/to/file.fit',
            'session_data': FITSessionData(
                threshold_power=250,
                avg_cadence=85,
                avg_temperature=22
            ),
            'lap_data': [
                FITLapData(avg_power=200, avg_cadence=85)
            ],
            'device_info': FITDeviceInfo(
                manufacturer='Garmin',
                product='Edge 830'
            ),
            'summary_stats': FITSummaryStats(
                power_avg=200,
                power_max=500,
                cadence_avg=85,
                hr_avg=150
            )
        }
        
        validated = FITParseResult(**parse_result)
        
        self.assertIsNotNone(validated.session_data)
        self.assertEqual(validated.session_data.threshold_power, 250)
        self.assertEqual(len(validated.lap_data), 1)
        self.assertIsNotNone(validated.device_info)
        
    def test_strict_activity_model_validation(self):
        """Test strict activity model with all FIT fields."""
        activity_data = {
            'id': 'test123',
            'name': 'Morning Ride',
            'type': 'Ride',
            'start_date_local': datetime.now(),
            'distance': 25000,
            'moving_time': 3600,
            # FIT-specific fields
            'threshold_power': 250,
            'functional_threshold_power': 250,
            'avg_cadence': 85.0,
            'max_cadence': 110.0,
            'avg_temperature': 22.0,
            'max_temperature': 28.0,
            'left_right_balance': 49.5,
            'avg_left_torque_effectiveness': 78.5,
            'avg_right_torque_effectiveness': 79.2,
            'avg_left_pedal_smoothness': 35.5,
            'avg_right_pedal_smoothness': 36.2,
            'total_training_effect': 3.2,
            'total_anaerobic_training_effect': 2.5,
            'intensity_factor': 0.88,
            'total_work': 720000,
            'device_manufacturer': 'Garmin',
            'device_product': 'Edge 830',
            'lap_data': json.dumps([{'lap': 1}]),
            'has_fit_data': True
        }
        
        validated = StrictActivityModel(**activity_data)
        
        # Verify ALL critical FIT fields are preserved
        self.assertEqual(validated.threshold_power, 250)
        self.assertEqual(validated.avg_cadence, 85.0)
        self.assertEqual(validated.avg_temperature, 22.0)
        self.assertEqual(validated.left_right_balance, 49.5)
        self.assertEqual(validated.device_manufacturer, 'Garmin')
        self.assertIsNotNone(validated.lap_data)
        
    def test_data_quality_enforcement(self):
        """Test that data quality is strictly enforced at 95% minimum."""
        # Create test dataframe with missing CORE data (not equipment-dependent)
        test_data = {
            'id': ['1', '2', '3', None, None],  # 60% populated - CORE field!
            'name': ['Ride 1', 'Ride 2', 'Ride 3', 'Ride 4', 'Ride 5'],
            'type': ['Ride', 'Ride', 'Ride', 'Ride', 'Ride'],
            'start_date_local': [datetime.now()] * 5,
            'distance': [25000, 30000, 20000, 25000, 28000],
            'moving_time': [3600, 4000, 3000, 3500, 3800],
            # Equipment-dependent fields can be missing - that's OK now
            'threshold_power': [250, 250, None, 250, 250],  # 80% populated
            'avg_cadence': [85, 87, None, 85, 86],  # 80% populated
        }
        
        df = pl.DataFrame(test_data)
        
        # Check quality metrics
        metrics = DataValidator.ensure_data_quality(df)
        
        # With missing CORE field (id), quality should be below 95%
        self.assertLess(metrics['quality_score'], 95, 
                       "Data quality should be below 95% with missing CORE fields")
        
    def test_fit_parser_with_real_file(self):
        """Test FIT parser with a real FIT file if available."""
        fit_path = Path("data/bronze/fit_files")
        if fit_path.exists():
            fit_files = list(fit_path.glob("*.fit"))
            if fit_files:
                # Parse first FIT file
                result = self.parser.parse_fit_file(fit_files[0])
                
                # Validate result with Pydantic
                self.assertIsInstance(result, dict)
                
                if 'session_data' in result:
                    # Validate session data
                    session = result['session_data']
                    if session:
                        validated_session = FITSessionData(**session)
                        
                        # Check critical fields
                        if validated_session.threshold_power:
                            self.assertIsInstance(validated_session.threshold_power, int)
                            self.assertGreater(validated_session.threshold_power, 0)
                        
                        if validated_session.avg_cadence:
                            self.assertIsInstance(validated_session.avg_cadence, int)
                            self.assertGreater(validated_session.avg_cadence, 0)
                
    def test_data_validator_rejects_low_quality(self):
        """Test that DataValidator rejects data below 95% quality."""
        # Create test data with ALL rows having null values in required fields
        test_data = {
            'id': ['1', '2', '3', '4'],
            'name': [None, None, None, None],  # 100% null - required field
            'type': [None, None, None, None],  # 100% null - required field
            'start_date_local': [None, None, None, None],  # 100% null - required field
            'distance': [25000, 30000, 20000, 25000],  # populated but doesn't matter
        }
        
        df = pl.DataFrame(test_data)
        
        # This should raise an error because all rows are invalid (missing required fields)
        with self.assertRaises(ValueError) as context:
            DataValidator.validate_dataframe(df)
        
        # Should fail because no rows can be validated
        self.assertIn("No valid activities", str(context.exception))
        
    def test_fit_data_completeness_check(self):
        """Test that we check for ALL expected FIT fields."""
        expected_fit_fields = [
            'threshold_power',
            'functional_threshold_power',
            'avg_cadence',
            'max_cadence',
            'avg_temperature',
            'max_temperature',
            'left_right_balance',
            'avg_left_torque_effectiveness',
            'avg_right_torque_effectiveness',
            'avg_left_pedal_smoothness',
            'avg_right_pedal_smoothness',
            'total_training_effect',
            'total_anaerobic_training_effect',
            'intensity_factor',
            'training_stress_score',
            'total_work',
            'device_manufacturer',
            'device_product',
            'lap_data',
            'zones_config'
        ]
        
        # Create activity with all fields
        complete_activity = {
            'id': 'test',
            'name': 'Test Ride',
            'type': 'Ride',
            'start_date_local': datetime.now(),
            'distance': 25000,
            'moving_time': 3600
        }
        
        # Add all FIT fields
        for field in expected_fit_fields:
            if 'lap' in field or 'zone' in field:
                complete_activity[field] = '{}'  # JSON string
            elif 'manufacturer' in field or 'product' in field:
                complete_activity[field] = 'Test'
            else:
                complete_activity[field] = 100
        
        # Should validate successfully
        validated = StrictActivityModel(**complete_activity)
        
        # Check all fields are present
        for field in expected_fit_fields:
            self.assertTrue(hasattr(validated, field), f"Missing field: {field}")
            
    def test_database_manager_quality_enforcement(self):
        """Test that DatabaseManager enforces 95% quality on save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = DatabaseManager(db_path=str(db_path))
            
            # Create low quality data
            poor_data = pl.DataFrame({
                'id': ['1', '2', '3'],
                'name': ['Ride 1', None, None],  # 33% populated
                'type': ['Ride', None, None],
                'start_date_local': [datetime.now().isoformat(), None, None],
                'distance': [25000, None, None]
            })
            
            # This should fail due to low quality
            with self.assertRaises(ValueError) as context:
                db.save_activities(poor_data)
            
            # Check for quality error message (33.3% is unacceptable)
            error_msg = str(context.exception).lower()
            self.assertTrue(
                "unacceptable" in error_msg and "95%" in error_msg,
                f"Expected quality error message, got: {context.exception}"
            )


def run_all_tests():
    """Run all data validation tests."""
    print("=" * 70)
    print("COMPREHENSIVE FIT DATA VALIDATION TESTS")
    print("MINIMUM ACCEPTABLE DATA QUALITY: 95%")
    print("=" * 70)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFITDataValidation)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED - Data validation is working correctly!")
    else:
        print(f"❌ TESTS FAILED - {len(result.failures)} failures, {len(result.errors)} errors")
        print("FIX THESE IMMEDIATELY! Data quality is NON-NEGOTIABLE!")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)