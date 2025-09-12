"""Module for merging FIT file data with activities ensuring ALL fields are captured."""

import json
import logging
from typing import Dict, Any, Optional
import polars as pl
from pathlib import Path

logger = logging.getLogger(__name__)


class FITDataMerger:
    """Handles merging of FIT file data with activity records."""
    
    @staticmethod
    def merge_fit_to_activity(activity_row: Dict[str, Any], fit_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge FIT file data into an activity row, ensuring ALL fields are captured.
        
        Args:
            activity_row: Original activity data
            fit_data: Parsed FIT file data
            
        Returns:
            Updated activity row with FIT data
        """
        if not fit_data:
            return activity_row
            
        # Create a copy to avoid modifying original
        updated = activity_row.copy()
        
        # Process session data
        if 'session_data' in fit_data and fit_data['session_data']:
            session = fit_data['session_data']
            
            # Critical power metrics that are currently missing
            if session.get('threshold_power'):
                updated['threshold_power'] = session['threshold_power']
            if session.get('functional_threshold_power'):
                updated['functional_threshold_power'] = session['functional_threshold_power']
            
            # Cadence metrics (fixing null issue)
            if session.get('avg_cadence') is not None:
                updated['avg_cadence'] = session['avg_cadence']
            if session.get('max_cadence') is not None:
                updated['max_cadence'] = session['max_cadence']
            if session.get('avg_fractional_cadence') is not None:
                updated['avg_fractional_cadence'] = session['avg_fractional_cadence']
            if session.get('max_fractional_cadence') is not None:
                updated['max_fractional_cadence'] = session['max_fractional_cadence']
            
            # Temperature data
            if session.get('avg_temperature') is not None:
                updated['avg_temperature'] = session['avg_temperature']
            if session.get('max_temperature') is not None:
                updated['max_temperature'] = session['max_temperature']
            
            # Advanced power metrics
            if session.get('left_right_balance') is not None:
                updated['left_right_balance'] = session['left_right_balance']
            if session.get('avg_left_torque_effectiveness') is not None:
                updated['avg_left_torque_effectiveness'] = session['avg_left_torque_effectiveness']
            if session.get('avg_right_torque_effectiveness') is not None:
                updated['avg_right_torque_effectiveness'] = session['avg_right_torque_effectiveness']
            if session.get('avg_left_pedal_smoothness') is not None:
                updated['avg_left_pedal_smoothness'] = session['avg_left_pedal_smoothness']
            if session.get('avg_right_pedal_smoothness') is not None:
                updated['avg_right_pedal_smoothness'] = session['avg_right_pedal_smoothness']
            
            # Power phase data (store as JSON strings)
            for phase_field in ['avg_left_power_phase', 'avg_right_power_phase', 
                              'avg_left_power_phase_peak', 'avg_right_power_phase_peak']:
                if session.get(phase_field):
                    updated[phase_field] = json.dumps(session[phase_field])
            
            # Training metrics
            if session.get('total_anaerobic_training_effect') is not None:
                updated['total_anaerobic_training_effect'] = session['total_anaerobic_training_effect']
            if session.get('total_training_effect') is not None:
                updated['total_training_effect'] = session['total_training_effect']
            if session.get('intensity_factor') is not None:
                updated['intensity_factor'] = session['intensity_factor']
            if session.get('training_stress_score') is not None:
                updated['training_stress_score'] = session['training_stress_score']
            if session.get('avg_vam') is not None:
                updated['avg_vam'] = session['avg_vam']
            if session.get('total_work') is not None:
                updated['total_work'] = session['total_work']
            
            # Update power/HR if available (don't override if already set)
            if session.get('avg_power') and not updated.get('average_power'):
                updated['average_power'] = session['avg_power']
            if session.get('max_power') and not updated.get('max_power'):
                updated['max_power'] = session['max_power']
            if session.get('normalized_power') and not updated.get('normalized_power'):
                updated['normalized_power'] = session['normalized_power']
            if session.get('avg_heart_rate') and not updated.get('average_heartrate'):
                updated['average_heartrate'] = session['avg_heart_rate']
            if session.get('max_heart_rate') and not updated.get('max_heartrate'):
                updated['max_heartrate'] = session['max_heart_rate']
        
        # Process device info
        if 'device_info' in fit_data and fit_data['device_info']:
            device = fit_data['device_info']
            if device.get('manufacturer'):
                updated['device_manufacturer'] = device['manufacturer']
            if device.get('product') or device.get('product_name'):
                updated['device_product'] = device.get('product') or device.get('product_name')
            if device.get('serial_number') is not None:
                updated['device_serial_number'] = str(device['serial_number'])
            if device.get('software_version') is not None:
                updated['device_software_version'] = device['software_version']
        
        # Store lap data as JSON
        if 'lap_data' in fit_data and fit_data['lap_data']:
            updated['lap_data'] = json.dumps(fit_data['lap_data'])
        
        # Store zones configuration
        if 'zones' in fit_data and fit_data['zones']:
            updated['zones_config'] = json.dumps(fit_data['zones'])
        
        # Mark that we have FIT analysis
        updated['has_fit_analysis'] = True
        
        return updated
    
    @staticmethod
    def merge_fit_to_dataframe(activities_df: pl.DataFrame, fit_data_map: Dict[str, Dict]) -> pl.DataFrame:
        """
        Merge FIT data for multiple activities into a dataframe.
        
        Args:
            activities_df: Activities dataframe
            fit_data_map: Map of activity_id -> fit_data
            
        Returns:
            Updated dataframe with FIT data
        """
        if not fit_data_map:
            return activities_df
        
        # Convert to dict of rows
        rows = activities_df.to_dicts()
        
        # Process each row
        updated_rows = []
        for row in rows:
            activity_id = row.get('garmin_activity_id') or row.get('id')
            if activity_id and str(activity_id) in fit_data_map:
                fit_data = fit_data_map[str(activity_id)]
                row = FITDataMerger.merge_fit_to_activity(row, fit_data)
            updated_rows.append(row)
        
        # Convert back to dataframe
        return pl.DataFrame(updated_rows)
    
    @staticmethod
    def validate_fit_integration(activities_df: pl.DataFrame) -> Dict[str, Any]:
        """
        Validate that FIT data is properly integrated.
        
        Args:
            activities_df: Activities dataframe to validate
            
        Returns:
            Validation report
        """
        report = {
            'total_activities': len(activities_df),
            'missing_fields': {},
            'data_quality': {}
        }
        
        # Check critical FIT fields
        fit_fields = [
            'threshold_power',
            'avg_cadence',
            'max_cadence',
            'avg_temperature',
            'left_right_balance',
            'avg_left_torque_effectiveness',
            'avg_right_torque_effectiveness',
            'total_work',
            'device_manufacturer',
            'lap_data'
        ]
        
        for field in fit_fields:
            if field in activities_df.columns:
                non_null = activities_df.filter(pl.col(field).is_not_null()).shape[0]
                null_pct = ((len(activities_df) - non_null) / len(activities_df)) * 100
                report['missing_fields'][field] = {
                    'null_count': len(activities_df) - non_null,
                    'null_percentage': null_pct,
                    'non_null_count': non_null
                }
            else:
                report['missing_fields'][field] = {
                    'null_count': len(activities_df),
                    'null_percentage': 100.0,
                    'non_null_count': 0,
                    'error': 'Field not in dataframe'
                }
        
        # Calculate overall data quality score
        total_checks = len(fit_fields) * len(activities_df)
        total_missing = sum(f['null_count'] for f in report['missing_fields'].values())
        report['data_quality']['fit_completeness'] = ((total_checks - total_missing) / total_checks) * 100
        
        # Check for FTP data specifically
        if 'threshold_power' in activities_df.columns:
            ftp_activities = activities_df.filter(pl.col('threshold_power').is_not_null())
            report['data_quality']['activities_with_ftp'] = len(ftp_activities)
            if len(ftp_activities) > 0:
                report['data_quality']['ftp_range'] = {
                    'min': ftp_activities['threshold_power'].min(),
                    'max': ftp_activities['threshold_power'].max(),
                    'mean': ftp_activities['threshold_power'].mean()
                }
        
        # Check cadence data
        if 'avg_cadence' in activities_df.columns:
            cadence_activities = activities_df.filter(pl.col('avg_cadence').is_not_null())
            report['data_quality']['activities_with_cadence'] = len(cadence_activities)
        
        return report


def test_fit_merger():
    """Test the FIT data merger with sample data."""
    
    # Sample activity
    activity = {
        'id': '12345',
        'name': 'Morning Ride',
        'distance': 25000,
        'moving_time': 3600
    }
    
    # Sample FIT data
    fit_data = {
        'session_data': {
            'threshold_power': 250,
            'avg_cadence': 85,
            'max_cadence': 110,
            'avg_temperature': 22,
            'left_right_balance': 49.5,
            'avg_left_torque_effectiveness': 78.5,
            'avg_right_torque_effectiveness': 79.2,
            'total_work': 900000
        },
        'device_info': {
            'manufacturer': 'Garmin',
            'product': 'Edge 830',
            'serial_number': 123456789,
            'software_version': 9.10
        },
        'lap_data': [
            {'lap_number': 1, 'avg_power': 245},
            {'lap_number': 2, 'avg_power': 255}
        ]
    }
    
    # Merge
    merger = FITDataMerger()
    updated = merger.merge_fit_to_activity(activity, fit_data)
    
    # Validate
    assert updated['threshold_power'] == 250
    assert updated['avg_cadence'] == 85
    assert updated['device_manufacturer'] == 'Garmin'
    assert 'lap_data' in updated
    
    print("✅ FIT merger test passed")
    print(f"Updated activity has {len(updated)} fields")
    
    return updated


if __name__ == "__main__":
    # Run test
    result = test_fit_merger()
    print("\nMerged activity data:")
    for key, value in result.items():
        print(f"  {key}: {value}")