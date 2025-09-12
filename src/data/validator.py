"""Strict data validation layer ensuring data accuracy and consistency."""

import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import polars as pl
import pandas as pd
from pydantic import BaseModel, ValidationError, Field, validator

logger = logging.getLogger(__name__)


class StrictActivityModel(BaseModel):
    """Strict Pydantic model for activity validation with comprehensive field mapping."""
    
    # Core identifiers (REQUIRED)
    id: str
    start_date_local: datetime
    name: str
    type: str
    
    # Time fields - handle multiple source formats
    moving_time: Optional[float] = None
    moving_duration: Optional[float] = None  # Garmin format
    elapsed_time: Optional[float] = None
    elapsed_duration: Optional[float] = None  # Garmin format
    duration: Optional[float] = None  # Another Garmin format
    
    # Distance
    distance: Optional[float] = None
    
    # Elevation - handle multiple formats
    total_elevation_gain: Optional[float] = None
    elevation_gain: Optional[float] = None  # Garmin format
    elevation_loss: Optional[float] = None
    
    # Speed metrics
    max_speed: Optional[float] = None
    average_speed: Optional[float] = None
    pace: Optional[float] = None
    threshold_pace: Optional[float] = None
    
    # Heart rate - handle multiple formats
    has_heartrate: Optional[bool] = None
    max_heartrate: Optional[int] = None
    max_hr: Optional[int] = None  # Garmin format
    average_heartrate: Optional[int] = None
    average_hr: Optional[int] = None  # Garmin format
    lthr: Optional[int] = None
    lactate_threshold_hr: Optional[float] = None  # Garmin format
    
    # Power data - handle multiple formats
    device_watts: Optional[bool] = None
    icu_average_watts: Optional[float] = None
    average_power: Optional[float] = None  # Garmin format
    icu_normalized_watts: Optional[float] = None
    normalized_power: Optional[float] = None  # Garmin format
    max_power: Optional[float] = None
    icu_joules: Optional[float] = None
    icu_intensity: Optional[float] = None
    intensity_factor: Optional[float] = None  # Garmin format
    icu_ftp: Optional[int] = None
    icu_eftp: Optional[float] = None
    icu_w_prime: Optional[float] = None
    icu_power_spike_threshold: Optional[float] = None
    icu_pm_ftp: Optional[int] = None
    icu_pm_cp: Optional[int] = None
    icu_pm_w_prime: Optional[float] = None
    icu_pm_p_max: Optional[int] = None
    
    # Training metrics - handle multiple formats
    icu_training_load: Optional[float] = None
    training_stress_score: Optional[float] = None  # Garmin TSS
    icu_training_load_edited: Optional[float] = None
    icu_fatigue: Optional[float] = None
    icu_fitness: Optional[float] = None
    icu_variability: Optional[float] = None
    icu_efficiency: Optional[float] = None
    icu_rpe: Optional[float] = None
    power_load: Optional[float] = None
    hr_load: Optional[float] = None
    pace_load: Optional[float] = None
    training_effect: Optional[float] = None  # Garmin format
    anaerobic_training_effect: Optional[float] = None  # Garmin format
    
    # Cadence
    average_cadence: Optional[float] = None
    max_cadence: Optional[float] = None
    
    # Other metrics
    calories: Optional[int] = None
    bmr_calories: Optional[int] = None  # Garmin format
    icu_weight: Optional[float] = None
    compliance: Optional[float] = None
    vo2_max: Optional[float] = None  # Garmin format
    
    # Flags
    trainer: Optional[bool] = None
    commute: Optional[bool] = None
    race: Optional[bool] = None
    icu_ignore_power: Optional[bool] = None
    icu_ignore_hr: Optional[bool] = None
    icu_ignore_time: Optional[bool] = None
    
    # Zone time in seconds (allow floats but convert to int)
    hr_z1_secs: Optional[float] = None
    hr_z2_secs: Optional[float] = None
    hr_z3_secs: Optional[float] = None
    hr_z4_secs: Optional[float] = None
    hr_z5_secs: Optional[float] = None
    hr_z6_secs: Optional[float] = None
    hr_z7_secs: Optional[float] = None
    
    z1_secs: Optional[float] = None
    z2_secs: Optional[float] = None
    z3_secs: Optional[float] = None
    z4_secs: Optional[float] = None
    z5_secs: Optional[float] = None
    z6_secs: Optional[float] = None
    z7_secs: Optional[float] = None
    sweet_spot_secs: Optional[float] = None
    
    # Additional Garmin fields
    pr_count: Optional[int] = None
    has_polyline: Optional[bool] = None
    has_splits: Optional[bool] = None
    sport_type: Optional[str] = None
    device_name: Optional[str] = None
    garmin_activity_id: Optional[Union[str, int]] = None
    
    # FIT-specific fields from session data
    threshold_power: Optional[float] = None  # FTP from FIT file
    functional_threshold_power: Optional[float] = None  # Alternative FTP field
    left_right_balance: Optional[float] = None
    avg_left_torque_effectiveness: Optional[float] = None
    avg_right_torque_effectiveness: Optional[float] = None
    avg_left_pedal_smoothness: Optional[float] = None
    avg_right_pedal_smoothness: Optional[float] = None
    avg_left_power_phase: Optional[Union[str, list]] = None  # JSON string or list
    avg_right_power_phase: Optional[Union[str, list]] = None
    avg_left_power_phase_peak: Optional[Union[str, list]] = None
    avg_right_power_phase_peak: Optional[Union[str, list]] = None
    
    # Cadence (fixing the issue where these are always null)
    avg_cadence: Optional[float] = None
    max_cadence: Optional[float] = None
    avg_fractional_cadence: Optional[float] = None
    max_fractional_cadence: Optional[float] = None
    
    # Advanced training metrics
    total_anaerobic_training_effect: Optional[float] = None
    total_training_effect: Optional[float] = None
    intensity_factor: Optional[float] = None
    avg_vam: Optional[float] = None  # Vertical Ascent Meters/hour
    
    # Work and energy
    total_work: Optional[int] = None  # Joules
    
    # Device info
    device_manufacturer: Optional[str] = None
    device_product: Optional[str] = None
    device_serial_number: Optional[str] = None
    device_software_version: Optional[float] = None
    
    # Location data
    start_latitude: Optional[float] = None
    start_longitude: Optional[float] = None
    end_latitude: Optional[float] = None
    end_longitude: Optional[float] = None
    
    # Temperature
    min_temperature: Optional[float] = None
    max_temperature: Optional[float] = None
    avg_temperature: Optional[float] = None
    
    # Metadata
    start_date: Optional[datetime] = None
    start_date_gmt: Optional[datetime] = None
    icu_sync_date: Optional[datetime] = None
    timezone: Optional[str] = None
    file_type: Optional[str] = None
    external_id: Optional[str] = None
    gear: Optional[str] = None
    description: Optional[str] = None
    source: Optional[str] = None
    
    # File tracking
    file_source: Optional[str] = None
    file_hash: Optional[str] = None
    import_timestamp: Optional[datetime] = None
    fit_file_path: Optional[str] = None
    
    # JSON fields (store as strings)
    hr_zones_data: Optional[Union[str, dict, list]] = None
    splits_data: Optional[Union[str, dict]] = None
    lap_data: Optional[Union[str, list]] = None  # Store lap data as JSON
    zones_config: Optional[Union[str, dict]] = None  # Zone configuration
    
    # Deduplication fields
    is_exact_duplicate: Optional[bool] = None
    is_potential_duplicate: Optional[bool] = None
    duplicate_count: Optional[int] = None
    
    class Config:
        """Pydantic config."""
        str_strip_whitespace = True
        # Allow both field names and aliases
        allow_population_by_field_name = True
        # Handle JSON serialization
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
    
    @validator('*', pre=True)
    def empty_str_to_none(cls, v):
        """Convert empty strings to None."""
        if isinstance(v, str) and v.strip() == '':
            return None
        return v
    
    @validator('start_date_local', 'start_date', 'start_date_gmt', 'icu_sync_date', pre=True)
    def parse_datetime(cls, v):
        """Parse datetime from various formats."""
        if v is None or pd.isna(v):
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace('Z', '+00:00'))
            except:
                try:
                    return pd.to_datetime(v)
                except:
                    return None
        return v
    
    @validator('hr_zones_data', 'splits_data', pre=True)
    def handle_json_fields(cls, v):
        """Handle JSON fields - convert to string if needed."""
        if v is None or pd.isna(v):
            return None
        if isinstance(v, (dict, list)):
            import json
            return json.dumps(v)
        return str(v)
    
    def normalize(self) -> Dict[str, Any]:
        """Normalize data by consolidating duplicate fields."""
        data = self.dict()
        
        # Consolidate time fields
        if not data.get('moving_time') and data.get('moving_duration'):
            data['moving_time'] = data['moving_duration']
        if not data.get('elapsed_time') and data.get('elapsed_duration'):
            data['elapsed_time'] = data['elapsed_duration']
        
        # Consolidate elevation
        if not data.get('total_elevation_gain') and data.get('elevation_gain'):
            data['total_elevation_gain'] = data['elevation_gain']
        
        # Consolidate heart rate
        if not data.get('average_heartrate') and data.get('average_hr'):
            data['average_heartrate'] = data['average_hr']
        if not data.get('max_heartrate') and data.get('max_hr'):
            data['max_heartrate'] = data['max_hr']
        
        # Consolidate power
        if not data.get('icu_average_watts') and data.get('average_power'):
            data['icu_average_watts'] = data['average_power']
        if not data.get('icu_normalized_watts') and data.get('normalized_power'):
            data['icu_normalized_watts'] = data['normalized_power']
        
        # Consolidate training load
        if not data.get('icu_training_load') and data.get('training_stress_score'):
            data['icu_training_load'] = data['training_stress_score']
        
        # Set flags based on data presence
        data['has_heartrate'] = bool(data.get('average_heartrate') or data.get('max_heartrate'))
        data['device_watts'] = bool(data.get('icu_average_watts') or data.get('icu_normalized_watts'))
        
        return data


class DataValidator:
    """Validates and normalizes activity data from any source."""
    
    @staticmethod
    def validate_activity(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize a single activity."""
        try:
            # Create model instance for validation
            activity = StrictActivityModel(**data)
            # Return normalized data
            return activity.normalize()
        except ValidationError as e:
            logger.error(f"Validation error for activity {data.get('id', 'unknown')}: {e}")
            raise
    
    @staticmethod
    def validate_dataframe(df: Union[pl.DataFrame, pd.DataFrame]) -> pl.DataFrame:
        """Validate and normalize entire dataframe."""
        # Convert to pandas for easier manipulation
        if isinstance(df, pl.DataFrame):
            df_pandas = df.to_pandas()
        else:
            df_pandas = df
        
        validated_rows = []
        errors = []
        
        for idx, row in df_pandas.iterrows():
            try:
                # Convert row to dict
                row_dict = row.to_dict()
                # Validate and normalize
                validated = DataValidator.validate_activity(row_dict)
                validated_rows.append(validated)
            except ValidationError as e:
                errors.append({
                    'index': idx,
                    'id': row.get('id', 'unknown'),
                    'error': str(e)
                })
        
        if errors:
            logger.warning(f"Validation errors for {len(errors)} activities:")
            for err in errors[:5]:  # Show first 5 errors
                logger.warning(f"  Activity {err['id']}: {err['error']}")
        
        # Convert validated data back to polars
        if validated_rows:
            return pl.DataFrame(validated_rows)
        else:
            raise ValueError("No valid activities found in dataframe")
    
    @staticmethod
    def ensure_data_quality(df: pl.DataFrame) -> Dict[str, Any]:
        """Check data quality and return metrics.
        
        Handles legitimate equipment variations:
        - Core fields (required): Must be 95%+ present
        - Equipment-dependent fields: Tracked but not required
        """
        metrics = {
            'total_rows': len(df),
            'columns': len(df.columns),
            'null_percentages': {},
            'data_types': {},
            'value_ranges': {},
            'quality_score': 0.0,
            'fit_data_coverage': {},
            'equipment_variations': {}
        }
        
        # CORE fields that MUST be present (legitimate data requirements)
        core_required_cols = [
            'id', 'start_date_local', 'name', 'type',
            'moving_time', 'distance'
        ]
        
        # EXPECTED fields (should be present if equipment supports)
        expected_cols = [
            'total_elevation_gain', 'average_heartrate'
        ]
        
        # OPTIONAL fields (equipment-dependent - power meter required)
        # Based on analysis: 83.5% of files have these (post March 2025)
        optional_power_fields = [
            'icu_average_watts', 'icu_training_load',
            'threshold_power', 'avg_cadence', 'avg_temperature',
            'left_right_balance', 'normalized_power'
        ]
        
        # Check CORE required fields (must meet 95% threshold)
        core_scores = []
        for col in core_required_cols:
            if col in df.columns:
                null_count = df[col].is_null().sum()
                null_pct = (null_count / len(df)) * 100
                metrics['null_percentages'][col] = null_pct
                core_scores.append(100 - null_pct)
            else:
                # Core field missing entirely - this is a problem
                metrics['null_percentages'][col] = 100
                core_scores.append(0)
        
        # Check EXPECTED fields (tracked but more lenient)
        expected_scores = []
        for col in expected_cols:
            if col in df.columns:
                null_count = df[col].is_null().sum()
                null_pct = (null_count / len(df)) * 100
                metrics['null_percentages'][col] = null_pct
                expected_scores.append(100 - null_pct)
        
        # Check OPTIONAL power meter fields (for information only)
        power_coverage = []
        for col in optional_power_fields:
            if col in df.columns:
                null_count = df[col].is_null().sum()
                null_pct = (null_count / len(df)) * 100
                metrics['fit_data_coverage'][col] = 100 - null_pct
                if 100 - null_pct > 0:  # Has some data
                    power_coverage.append(100 - null_pct)
        
        # Detect equipment variation
        if power_coverage:
            avg_power_coverage = sum(power_coverage) / len(power_coverage)
            if avg_power_coverage > 80:
                metrics['equipment_variations']['has_power_meter'] = True
                metrics['equipment_variations']['power_data_coverage'] = f'{avg_power_coverage:.1f}%'
            elif avg_power_coverage > 0:
                metrics['equipment_variations']['partial_power_data'] = True
                metrics['equipment_variations']['power_data_coverage'] = f'{avg_power_coverage:.1f}%'
        
        # Check data types
        for col in df.columns:
            metrics['data_types'][col] = str(df[col].dtype)
        
        # Check value ranges for numeric columns
        numeric_cols = df.select(pl.selectors.numeric()).columns
        for col in numeric_cols[:10]:  # Limit to first 10
            if not df[col].is_null().all():
                metrics['value_ranges'][col] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean()
                }
        
        # Calculate quality score based on CORE fields only
        # This reflects true data quality, not equipment limitations
        if core_scores:
            metrics['quality_score'] = sum(core_scores) / len(core_scores)
        
        # Track expected field quality separately
        if expected_scores:
            metrics['expected_field_score'] = sum(expected_scores) / len(expected_scores)
        
        # Track power meter data coverage (informational)
        if power_coverage:
            metrics['power_data_score'] = sum(power_coverage) / len(power_coverage)
        
        return metrics