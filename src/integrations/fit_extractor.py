"""Direct FIT file extractor that bypasses problematic parsing issues."""

import fitparse
from pathlib import Path
import polars as pl
from datetime import datetime
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DirectFITExtractor:
    """Extract FIT data directly using fitparse, handling all edge cases."""
    
    def __init__(self, fit_dir: Optional[Path] = None):
        """Initialize extractor.
        
        Args:
            fit_dir: Directory containing FIT files
        """
        self.fit_dir = fit_dir or Path("data/bronze/fit_files")
        
    def extract_all_fit_files(self) -> pl.DataFrame:
        """Extract data from all FIT files in directory.
        
        Returns:
            DataFrame with all extracted FIT data
        """
        fit_files = sorted(self.fit_dir.glob("*.fit"))
        logger.info(f"Found {len(fit_files)} FIT files to extract")
        
        all_activities = []
        errors = []
        
        for fit_path in fit_files:
            try:
                activity = self.extract_fit_file(fit_path)
                if activity:
                    all_activities.append(activity)
            except Exception as e:
                errors.append((fit_path.name, str(e)))
                logger.warning(f"Error extracting {fit_path.name}: {e}")
        
        if errors:
            logger.warning(f"Failed to extract {len(errors)} files")
            for filename, error in errors[:5]:  # Show first 5 errors
                logger.warning(f"  {filename}: {error}")
        
        if not all_activities:
            logger.error("No activities extracted")
            return pl.DataFrame()
        
        # Create DataFrame
        df = pl.DataFrame(all_activities)
        logger.info(f"Successfully extracted {len(df)} activities with {len(df.columns)} fields")
        
        return df
    
    def extract_fit_file(self, fit_path: Path) -> Dict:
        """Extract all data from a single FIT file.
        
        Args:
            fit_path: Path to FIT file
            
        Returns:
            Dictionary with extracted data
        """
        activity_id = fit_path.stem.split('_')[0]
        
        fitfile = fitparse.FitFile(str(fit_path))
        
        # Initialize activity with required fields
        activity = {
            'id': f'garmin_{activity_id}',
            'garmin_activity_id': activity_id,
            'name': ' '.join(fit_path.stem.split('_')[1:]) if '_' in fit_path.stem else fit_path.stem,
            'type': 'Ride',
            'fit_file_path': str(fit_path),
        }
        
        # Extract session data (summary statistics)
        for record in fitfile.get_messages('session'):
            for field in record:
                if field.value is not None:
                    # Handle datetime fields
                    if 'time' in field.name or 'timestamp' in field.name:
                        if hasattr(field.value, 'isoformat'):
                            activity[field.name] = field.value.isoformat()
                        else:
                            activity[field.name] = str(field.value)
                    # Handle the left_right_balance bug
                    elif field.name == 'left_right_balance':
                        if isinstance(field.value, str):
                            # fitparse bug - ignore string values
                            continue
                        else:
                            # Convert from 128-228 scale to 0-100
                            if field.value >= 128:
                                activity[field.name] = field.value - 128
                            else:
                                activity[field.name] = field.value
                    # Store all other fields
                    else:
                        activity[field.name] = field.value
        
        # Extract device info
        for record in fitfile.get_messages('device_info'):
            for field in record:
                if field.value is not None and field.name:
                    activity[f'device_{field.name}'] = field.value
                    
        # Extract first lap data (if available)
        lap_count = 0
        for record in fitfile.get_messages('lap'):
            lap_count += 1
            if lap_count == 1:  # Just store first lap summary
                for field in record:
                    if field.value is not None and field.name:
                        # Avoid overwriting session data
                        if field.name not in activity:
                            activity[f'lap1_{field.name}'] = field.value
        
        activity['num_laps'] = lap_count
        
        # Ensure required fields have defaults
        if 'start_date_local' not in activity:
            activity['start_date_local'] = activity.get('timestamp', activity.get('start_time', datetime.now().isoformat()))
        if 'moving_time' not in activity:
            activity['moving_time'] = activity.get('total_timer_time', activity.get('total_elapsed_time', 0))
        if 'distance' not in activity:
            activity['distance'] = activity.get('total_distance', 0)
        
        # Mark as having FIT data
        activity['has_fit_data'] = True
        
        return activity
    
    def get_fit_coverage_report(self, df: pl.DataFrame) -> Dict:
        """Generate coverage report for FIT data.
        
        Args:
            df: DataFrame with FIT data
            
        Returns:
            Coverage statistics
        """
        report = {
            'total_activities': len(df),
            'field_coverage': {}
        }
        
        # Key FIT fields to check
        key_fields = [
            'threshold_power',
            'avg_cadence',
            'avg_temperature',
            'left_right_balance',
            'normalized_power',
            'training_stress_score',
            'intensity_factor',
            'total_work'
        ]
        
        for field in key_fields:
            if field in df.columns:
                non_null = df[field].is_not_null().sum()
                report['field_coverage'][field] = {
                    'count': non_null,
                    'percentage': (non_null / len(df)) * 100
                }
        
        # Overall FIT completeness
        if report['field_coverage']:
            avg_coverage = sum(f['percentage'] for f in report['field_coverage'].values()) / len(report['field_coverage'])
            report['overall_fit_coverage'] = avg_coverage
        
        return report