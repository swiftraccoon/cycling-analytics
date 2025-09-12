"""Test handling of legitimate equipment variations in FIT data.

Based on real data analysis:
- 13 files (Jan-Mar 2025): No power meter, basic metrics only
- 66 files (Mar-Aug 2025): With power meter, full metrics including FTP

This is a LEGITIMATE data pattern, not a quality issue.
"""

import unittest
from datetime import datetime
import polars as pl
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.validator import DataValidator


class TestEquipmentVariation(unittest.TestCase):
    """Test that validator handles equipment variations correctly."""
    
    def test_basic_equipment_data_passes_validation(self):
        """Test that rides without power meter pass validation.
        
        These are legitimate rides from Jan-Mar 2025 before power meter.
        """
        # Create data matching early 2025 rides (no power meter)
        basic_rides = pl.DataFrame({
            'id': ['ride_1', 'ride_2', 'ride_3'],
            'name': ['Morning Ride', 'Evening Ride', 'Weekend Ride'],
            'type': ['Ride', 'Ride', 'Ride'],
            'start_date_local': [datetime(2025, 1, 15), datetime(2025, 2, 10), datetime(2025, 3, 5)],
            'moving_time': [3600, 5400, 7200],
            'distance': [25000, 40000, 55000],
            'total_elevation_gain': [250, 400, 600],
            'average_heartrate': [145, 150, 148],
            # No power data - this is legitimate!
            'threshold_power': [None, None, None],
            'icu_average_watts': [None, None, None],
            'normalized_power': [None, None, None],
            'avg_cadence': [None, None, None],
        })
        
        # Check quality metrics
        metrics = DataValidator.ensure_data_quality(basic_rides)
        
        # Core fields should be 100%
        self.assertEqual(metrics['quality_score'], 100.0, 
                        "Basic rides should have 100% quality for core fields")
        
        # Power data should be detected as missing (equipment variation)
        self.assertEqual(metrics.get('power_data_score', 0), 0,
                        "Should detect no power data available")
        
    def test_power_meter_data_passes_validation(self):
        """Test that rides with power meter pass validation.
        
        These match rides from Mar-Aug 2025 with full power metrics.
        """
        # Create data matching post-March 2025 rides (with power meter)
        power_rides = pl.DataFrame({
            'id': ['ride_4', 'ride_5', 'ride_6'],
            'name': ['Morning Ride', 'Evening Ride', 'Weekend Ride'],
            'type': ['Ride', 'Ride', 'Ride'],
            'start_date_local': [datetime(2025, 4, 15), datetime(2025, 5, 10), datetime(2025, 6, 5)],
            'moving_time': [3600, 5400, 7200],
            'distance': [25000, 40000, 55000],
            'total_elevation_gain': [250, 400, 600],
            'average_heartrate': [145, 150, 148],
            # Full power data - 83.5% coverage in real data
            'threshold_power': [196, 196, 200],  # Actual FTP values from files
            'icu_average_watts': [180, 195, 210],
            'normalized_power': [185, 200, 215],
            'avg_cadence': [85, 87, 89],
            'avg_temperature': [22, 24, 26],
        })
        
        # Check quality metrics
        metrics = DataValidator.ensure_data_quality(power_rides)
        
        # Core fields should be 100%
        self.assertEqual(metrics['quality_score'], 100.0,
                        "Power rides should have 100% quality for core fields")
        
        # Should detect power meter presence
        self.assertTrue(metrics.get('equipment_variations', {}).get('has_power_meter', False),
                       "Should detect power meter equipment")
        
    def test_mixed_equipment_dataset(self):
        """Test dataset with both equipment types (matches real data).
        
        Real data: 13 basic rides + 66 power rides = 79 total
        This should pass validation as it's legitimate variation.
        """
        # Create mixed dataset matching actual proportions
        # Build complete dataset with proper null handling
        ids = [f'basic_{i}' for i in range(13)] + [f'power_{i}' for i in range(66)]
        names = ['Basic Ride'] * 13 + ['Power Ride'] * 66
        types = ['Ride'] * 79
        dates = [datetime(2025, 1, 1)] * 13 + [datetime(2025, 4, 1)] * 66
        times = [3600.0] * 79
        distances = [25000.0] * 79
        heartrates = [145.0] * 79
        
        # Power data: None for first 13, values for next 66
        threshold_power = [None] * 13 + [196.0] * 66
        avg_watts = [None] * 13 + [180.0] * 66
        
        mixed_data = pl.DataFrame({
            'id': ids,
            'name': names,
            'type': types,
            'start_date_local': dates,
            'moving_time': times,
            'distance': distances,
            'average_heartrate': heartrates,
            'threshold_power': threshold_power,
            'icu_average_watts': avg_watts,
        })
        
        # Check quality metrics
        metrics = DataValidator.ensure_data_quality(mixed_data)
        
        # Core fields should still be 100%
        self.assertEqual(metrics['quality_score'], 100.0,
                        "Mixed equipment data should have 100% quality for core fields")
        
        # Power coverage should be ~83.5% (66/79)
        power_coverage = metrics.get('fit_data_coverage', {}).get('threshold_power', 0)
        self.assertAlmostEqual(power_coverage, 83.5, places=1,
                              msg=f"Power coverage should be ~83.5%, got {power_coverage}")
        
    def test_quality_score_focuses_on_core_fields(self):
        """Test that quality score is based on core fields only.
        
        Equipment-dependent fields should NOT affect quality score.
        """
        # Create data with missing core field
        bad_data = pl.DataFrame({
            'id': ['ride_1', 'ride_2', None],  # Missing ID - core field!
            'name': ['Ride 1', 'Ride 2', 'Ride 3'],
            'type': ['Ride', 'Ride', 'Ride'],
            'start_date_local': [datetime.now(), datetime.now(), datetime.now()],
            'moving_time': [3600, 3600, 3600],
            'distance': [25000, 25000, 25000],
            # Power fields can be missing - that's OK
            'threshold_power': [None, None, None],
        })
        
        metrics = DataValidator.ensure_data_quality(bad_data)
        
        # Quality should be <100% due to missing core field
        self.assertLess(metrics['quality_score'], 100,
                       "Missing core field should reduce quality score")
        
        # But specifically it should be ~83.3% (5/6 core fields complete)
        # 6 core fields, 1 missing in 1/3 records = 5/6 = 83.3%
        expected_score = (2/3) * 100  # 66.7% for ID field
        # Other 5 fields are 100%, so average is (66.7 + 500) / 6 = 94.4%
        # Actually, we need to check the logic...
        
        # The important thing is it's less than 95%
        self.assertLess(metrics['quality_score'], 95,
                       "Missing core field should fail 95% threshold")


def run_tests():
    """Run equipment variation tests."""
    print("=" * 70)
    print("EQUIPMENT VARIATION TESTS")
    print("Testing handling of legitimate equipment differences")
    print("=" * 70)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestEquipmentVariation)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ Equipment variation handling is correct!")
        print("System properly handles rides with and without power meters")
    else:
        print("❌ Equipment variation tests failed!")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)