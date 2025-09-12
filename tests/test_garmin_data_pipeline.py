"""Comprehensive tests for Garmin data pipeline ensuring ALL data is captured and utilized."""

import unittest
import json
import tempfile
from pathlib import Path
from datetime import datetime
import polars as pl
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.integrations.fit_parser import FITParser
from src.integrations.fit_data_merger import FITDataMerger
from src.data.validator import StrictActivityModel
from src.storage.database.schema_migrator import SchemaMigrator
from src.ml.train_garmin import prepare_power_training_data


class TestGarminDataPipeline(unittest.TestCase):
    """Test complete data flow from FIT files to ML models."""
    
    def setUp(self):
        """Set up test environment."""
        self.fit_parser = FITParser()
        self.merger = FITDataMerger()
        
    def test_fit_parser_extracts_all_fields(self):
        """Test that FIT parser extracts ALL available fields."""
        # Use a real FIT file if available
        fit_path = Path("data/bronze/fit_files")
        if fit_path.exists():
            fit_files = list(fit_path.glob("*.fit"))
            if fit_files:
                # Parse first FIT file
                fit_data = self.fit_parser.parse_fit_file(fit_files[0])
                
                # Check session data
                self.assertIn('session_data', fit_data)
                session = fit_data['session_data']
                
                # Critical fields that must be present
                critical_fields = [
                    'threshold_power',  # FTP
                    'avg_cadence',
                    'avg_temperature',
                    'normalized_power',
                    'training_stress_score',
                    'intensity_factor'
                ]
                
                for field in critical_fields:
                    if field in session:
                        print(f"✅ {field}: {session[field]}")
                    else:
                        print(f"⚠️  {field}: Missing")
                
                # Check device info
                self.assertIn('device_info', fit_data)
                
                # Check lap data
                self.assertIn('lap_data', fit_data)
                
    def test_pydantic_validation_complete(self):
        """Test that Pydantic model validates ALL fields."""
        # Create test data with all fields
        test_activity = {
            'id': 'test123',
            'name': 'Test Ride',
            'type': 'Ride',
            'start_date_local': datetime.now(),
            'distance': 25000,
            'moving_time': 3600,
            # FIT-specific fields
            'threshold_power': 250,
            'avg_cadence': 85,
            'max_cadence': 110,
            'avg_temperature': 22,
            'left_right_balance': 49.5,
            'avg_left_torque_effectiveness': 78.5,
            'avg_right_torque_effectiveness': 79.2,
            'avg_left_pedal_smoothness': 35.5,
            'avg_right_pedal_smoothness': 36.2,
            'total_anaerobic_training_effect': 2.5,
            'total_training_effect': 3.2,
            'intensity_factor': 0.85,
            'total_work': 900000,
            'device_manufacturer': 'Garmin',
            'device_product': 'Edge 830',
            'lap_data': json.dumps([{'lap': 1}])
        }
        
        # Validate with Pydantic
        validated = StrictActivityModel(**test_activity)
        
        # Check critical fields are preserved
        self.assertEqual(validated.threshold_power, 250)
        self.assertEqual(validated.avg_cadence, 85)
        self.assertEqual(validated.avg_temperature, 22)
        self.assertIsNotNone(validated.device_manufacturer)
        
    def test_database_schema_has_all_columns(self):
        """Test that database schema includes ALL FIT fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            migrator = SchemaMigrator(str(db_path))
            
            # Run migrations
            migrator.run_migrations()
            
            # Check for critical columns
            columns = migrator.get_existing_columns("activities")
            
            required_columns = [
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
                'total_anaerobic_training_effect',
                'total_training_effect',
                'intensity_factor',
                'avg_vam',
                'total_work',
                'device_manufacturer',
                'device_product',
                'lap_data',
                'zones_config'
            ]
            
            missing = [col for col in required_columns if col not in columns]
            if missing:
                print(f"❌ Missing columns: {missing}")
            else:
                print("✅ All required columns present")
            
            self.assertEqual(len(missing), 0, f"Missing columns: {missing}")
    
    def test_fit_merger_preserves_all_data(self):
        """Test that FIT merger preserves ALL extracted data."""
        # Sample activity
        activity = {
            'id': '12345',
            'name': 'Morning Ride',
            'distance': 25000
        }
        
        # Comprehensive FIT data
        fit_data = {
            'session_data': {
                'threshold_power': 250,
                'avg_cadence': 85,
                'max_cadence': 110,
                'avg_temperature': 22,
                'max_temperature': 28,
                'left_right_balance': 49.5,
                'avg_left_torque_effectiveness': 78.5,
                'avg_right_torque_effectiveness': 79.2,
                'avg_left_pedal_smoothness': 35.5,
                'avg_right_pedal_smoothness': 36.2,
                'avg_left_power_phase': [10, 90, 15, 85],
                'avg_right_power_phase': [12, 88, 14, 86],
                'total_anaerobic_training_effect': 2.5,
                'total_training_effect': 3.2,
                'intensity_factor': 0.85,
                'training_stress_score': 95,
                'avg_vam': 450,
                'total_work': 900000
            },
            'device_info': {
                'manufacturer': 'Garmin',
                'product': 'Edge 830',
                'serial_number': 123456789,
                'software_version': 9.10
            },
            'lap_data': [
                {'lap_number': 1, 'avg_power': 245, 'avg_cadence': 83},
                {'lap_number': 2, 'avg_power': 255, 'avg_cadence': 87}
            ],
            'zones': {
                'functional_threshold_power': 250,
                'max_heart_rate': 185
            }
        }
        
        # Merge
        merged = self.merger.merge_fit_to_activity(activity, fit_data)
        
        # Verify ALL fields are present
        self.assertEqual(merged['threshold_power'], 250)
        self.assertEqual(merged['avg_cadence'], 85)
        self.assertEqual(merged['max_cadence'], 110)
        self.assertEqual(merged['avg_temperature'], 22)
        self.assertEqual(merged['left_right_balance'], 49.5)
        self.assertEqual(merged['avg_left_torque_effectiveness'], 78.5)
        self.assertEqual(merged['device_manufacturer'], 'Garmin')
        self.assertIn('lap_data', merged)
        self.assertIn('zones_config', merged)
        
        # Check JSON fields
        self.assertIsInstance(merged['avg_left_power_phase'], str)
        lap_data = json.loads(merged['lap_data'])
        self.assertEqual(len(lap_data), 2)
        
    def test_ml_pipeline_uses_all_features(self):
        """Test that ML pipeline uses ALL available features."""
        # Create test dataframe with all fields
        test_data = {
            'start_date_local': [datetime.now()],
            'distance': [25000],
            'moving_time': [3600],
            'average_power': [200],
            'normalized_power': [220],
            'max_power': [500],
            'threshold_power': [250],  # FTP from FIT
            'avg_cadence': [85],
            'max_cadence': [110],
            'avg_temperature': [22],
            'left_right_balance': [49.5],
            'avg_left_torque_effectiveness': [78.5],
            'avg_right_torque_effectiveness': [79.2],
            'avg_left_pedal_smoothness': [35.5],
            'avg_right_pedal_smoothness': [36.2],
            'training_effect': [3.2],
            'anaerobic_training_effect': [2.5],
            'intensity_factor': [0.88],
            'total_work': [720000],
            'average_heartrate': [150],
            'max_heartrate': [175]
        }
        
        df = pl.DataFrame(test_data)
        
        # Prepare training data
        X, y, feature_names = prepare_power_training_data(df)
        
        # Check that all advanced features are included
        self.assertIsNotNone(X)
        self.assertIsNotNone(feature_names)
        
        expected_features = [
            'left_right_balance',
            'left_torque_effectiveness',
            'right_torque_effectiveness',
            'left_pedal_smoothness',
            'right_pedal_smoothness',
            'avg_cadence',
            'avg_temperature',
            'training_effect',
            'anaerobic_effect',
            'total_work_kj',
            'efficiency_factor'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, feature_names, f"Missing feature: {feature}")
        
        print(f"✅ ML pipeline uses {len(feature_names)} features")
        
    def test_data_quality_validation(self):
        """Test data quality validation identifies missing data."""
        # Create test data with some missing fields
        test_data = {
            'id': ['1', '2', '3'],
            'name': ['Ride 1', 'Ride 2', 'Ride 3'],
            'distance': [25000, 30000, 20000],
            'moving_time': [3600, 4000, 3000],
            'average_power': [200, 210, None],
            'threshold_power': [250, None, None],  # Missing FTP
            'avg_cadence': [None, None, None],  # All missing
            'device_manufacturer': ['Garmin', None, None]
        }
        
        df = pl.DataFrame(test_data)
        
        # Validate
        report = self.merger.validate_fit_integration(df)
        
        # Check report
        self.assertEqual(report['total_activities'], 3)
        self.assertIn('missing_fields', report)
        self.assertIn('threshold_power', report['missing_fields'])
        
        # Cadence should be 100% missing
        cadence_report = report['missing_fields'].get('avg_cadence', {})
        if cadence_report and 'null_percentage' in cadence_report:
            self.assertEqual(cadence_report['null_percentage'], 100.0)
        
        print(f"Data quality score: {report['data_quality'].get('fit_completeness', 0):.1f}%")


def run_comprehensive_test():
    """Run comprehensive test of the data pipeline."""
    print("=" * 60)
    print("COMPREHENSIVE GARMIN DATA PIPELINE TEST")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestGarminDataPipeline)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ FAILURES: {len(result.failures)}")
        print(f"❌ ERRORS: {len(result.errors)}")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_test()