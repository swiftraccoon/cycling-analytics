"""Pydantic models for FIT file data structures with comprehensive validation."""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import polars as pl
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator, field_validator
import numpy as np


class PowerZone(str, Enum):
    """Power training zones based on FTP percentage."""
    RECOVERY = "zone1"  # < 55% FTP
    ENDURANCE = "zone2"  # 56-75% FTP
    TEMPO = "zone3"  # 76-90% FTP
    THRESHOLD = "zone4"  # 91-105% FTP
    VO2MAX = "zone5"  # 106-120% FTP
    ANAEROBIC = "zone6"  # 121-150% FTP
    NEUROMUSCULAR = "zone7"  # > 150% FTP


class HeartRateZone(str, Enum):
    """Heart rate zones based on max HR percentage."""
    ZONE1 = "zone1"  # < 60% max HR
    ZONE2 = "zone2"  # 60-70% max HR
    ZONE3 = "zone3"  # 70-80% max HR
    ZONE4 = "zone4"  # 80-90% max HR
    ZONE5 = "zone5"  # > 90% max HR


class FITDeviceInfo(BaseModel):
    """Device information from FIT file."""
    model_config = ConfigDict(extra='allow')
    
    manufacturer: Optional[str] = None
    product: Optional[str] = None
    serial_number: Optional[Union[str, int]] = None
    software_version: Optional[float] = None
    hardware_version: Optional[float] = None
    battery_voltage: Optional[float] = None
    device_index: Optional[int] = None
    device_type: Optional[str] = None
    product_name: Optional[str] = None
    
    @field_validator('serial_number')
    @classmethod
    def convert_serial(cls, v):
        """Convert serial number to string."""
        if v is not None:
            return str(v)
        return v


class FITSessionData(BaseModel):
    """Session summary data from FIT file."""
    model_config = ConfigDict(extra='allow')
    
    @field_validator('start_position_lat', 'start_position_long', 'end_position_lat', 'end_position_long', mode='before')
    @classmethod
    def convert_semicircles(cls, v):
        """Convert semicircles to degrees if needed."""
        if v is not None and abs(v) > 180:
            # Value is in semicircles, convert to degrees
            return v * (180.0 / 2147483648.0)
        return v
    
    @field_validator('left_right_balance', mode='before')
    @classmethod
    def normalize_balance(cls, v):
        """Normalize balance value if stored as scaled."""
        if v is not None and v > 100:
            # Value is scaled (e.g., 38017 = 38.017%)
            return v / 1000.0 if v > 1000 else v / 100.0
        return v
    
    # Time fields
    timestamp: Optional[datetime] = None
    start_time: Optional[datetime] = None
    total_elapsed_time: Optional[float] = Field(None, ge=0)
    total_timer_time: Optional[float] = Field(None, ge=0)
    
    # Distance and speed
    total_distance: Optional[float] = Field(None, ge=0)
    enhanced_avg_speed: Optional[float] = Field(None, ge=0)
    enhanced_max_speed: Optional[float] = Field(None, ge=0)
    avg_speed: Optional[float] = Field(None, ge=0)
    max_speed: Optional[float] = Field(None, ge=0)
    
    # Heart rate
    avg_heart_rate: Optional[int] = Field(None, ge=0, le=255)
    max_heart_rate: Optional[int] = Field(None, ge=0, le=255)
    min_heart_rate: Optional[int] = Field(None, ge=0, le=255)
    
    # Power
    avg_power: Optional[int] = Field(None, ge=0)
    max_power: Optional[int] = Field(None, ge=0)
    normalized_power: Optional[int] = Field(None, ge=0)
    left_right_balance: Optional[float] = Field(None)  # Can be scaled in FIT
    avg_left_torque_effectiveness: Optional[float] = Field(None, ge=0, le=100)
    avg_right_torque_effectiveness: Optional[float] = Field(None, ge=0, le=100)
    avg_left_pedal_smoothness: Optional[float] = Field(None, ge=0, le=100)
    avg_right_pedal_smoothness: Optional[float] = Field(None, ge=0, le=100)
    threshold_power: Optional[int] = Field(None, ge=0)
    
    # Cadence
    avg_cadence: Optional[int] = Field(None, ge=0, le=255)
    max_cadence: Optional[int] = Field(None, ge=0, le=255)
    avg_fractional_cadence: Optional[float] = Field(None, ge=0)
    max_fractional_cadence: Optional[float] = Field(None, ge=0)
    
    # Altitude
    total_ascent: Optional[float] = Field(None, ge=0)
    total_descent: Optional[float] = Field(None, ge=0)
    avg_altitude: Optional[float] = None
    max_altitude: Optional[float] = None
    min_altitude: Optional[float] = None
    
    # Temperature
    avg_temperature: Optional[int] = Field(None, ge=-100, le=100)
    max_temperature: Optional[int] = Field(None, ge=-100, le=100)
    
    # Calories and work
    total_calories: Optional[int] = Field(None, ge=0)
    total_work: Optional[int] = Field(None, ge=0)  # Joules
    
    # Training metrics
    training_stress_score: Optional[float] = Field(None, ge=0)
    intensity_factor: Optional[float] = Field(None, ge=0)
    
    # GPS
    start_position_lat: Optional[float] = Field(None, ge=-90, le=90)
    start_position_long: Optional[float] = Field(None, ge=-180, le=180)
    end_position_lat: Optional[float] = Field(None, ge=-90, le=90)
    end_position_long: Optional[float] = Field(None, ge=-180, le=180)
    
    # Sport info
    sport: Optional[str] = None
    sub_sport: Optional[str] = None
    
    # Message index
    message_index: Optional[int] = None
    
    @field_validator('timestamp', 'start_time')
    @classmethod
    def validate_datetime(cls, v):
        """Ensure datetime is timezone aware."""
        if v and not v.tzinfo:
            from datetime import timezone
            return v.replace(tzinfo=timezone.utc)
        return v


class FITLapData(BaseModel):
    """Lap data from FIT file."""
    model_config = ConfigDict(extra='allow')
    
    @field_validator('start_position_lat', 'start_position_long', 'end_position_lat', 'end_position_long', mode='before')
    @classmethod
    def convert_semicircles(cls, v):
        """Convert semicircles to degrees if needed."""
        if v is not None and abs(v) > 180:
            # Value is in semicircles, convert to degrees
            return v * (180.0 / 2147483648.0)
        return v
    
    timestamp: Optional[datetime] = None
    start_time: Optional[datetime] = None
    lap_trigger: Optional[str] = None
    
    # Time and distance
    total_elapsed_time: Optional[float] = Field(None, ge=0)
    total_timer_time: Optional[float] = Field(None, ge=0)
    total_distance: Optional[float] = Field(None, ge=0)
    
    # Speed
    enhanced_avg_speed: Optional[float] = Field(None, ge=0)
    enhanced_max_speed: Optional[float] = Field(None, ge=0)
    
    # Heart rate
    avg_heart_rate: Optional[int] = Field(None, ge=0, le=255)
    max_heart_rate: Optional[int] = Field(None, ge=0, le=255)
    
    # Power
    avg_power: Optional[int] = Field(None, ge=0)
    max_power: Optional[int] = Field(None, ge=0)
    normalized_power: Optional[int] = Field(None, ge=0)
    
    # Cadence
    avg_cadence: Optional[int] = Field(None, ge=0, le=255)
    max_cadence: Optional[int] = Field(None, ge=0, le=255)
    
    # Altitude
    total_ascent: Optional[float] = Field(None, ge=0)
    total_descent: Optional[float] = Field(None, ge=0)
    
    # Calories
    total_calories: Optional[int] = Field(None, ge=0)
    
    # Positions
    start_position_lat: Optional[float] = Field(None, ge=-90, le=90)
    start_position_long: Optional[float] = Field(None, ge=-180, le=180)
    end_position_lat: Optional[float] = Field(None, ge=-90, le=90)
    end_position_long: Optional[float] = Field(None, ge=-180, le=180)
    
    message_index: Optional[int] = None


class FITRecordData(BaseModel):
    """Individual record (data point) from FIT file."""
    model_config = ConfigDict(extra='allow')
    
    @field_validator('position_lat', 'position_long', mode='before')
    @classmethod
    def convert_semicircles(cls, v):
        """Convert semicircles to degrees if needed."""
        if v is not None and abs(v) > 180:
            # Value is in semicircles, convert to degrees
            return v * (180.0 / 2147483648.0)
        return v
    
    @field_validator('left_right_balance', mode='before')
    @classmethod
    def normalize_balance(cls, v):
        """Normalize balance value if stored as scaled."""
        if v is not None and v > 100:
            # Value is scaled
            return v / 1000.0 if v > 1000 else v / 100.0
        return v
    
    timestamp: datetime
    
    # Position
    position_lat: Optional[float] = Field(None, ge=-90, le=90)
    position_long: Optional[float] = Field(None, ge=-180, le=180)
    
    # Distance and speed
    distance: Optional[float] = Field(None, ge=0)  # meters
    enhanced_speed: Optional[float] = Field(None, ge=0)  # m/s
    speed: Optional[float] = Field(None, ge=0)  # m/s
    
    # Altitude
    enhanced_altitude: Optional[float] = None  # meters
    altitude: Optional[float] = None  # meters
    
    # Heart rate
    heart_rate: Optional[int] = Field(None, ge=0, le=255)  # bpm
    
    # Cadence
    cadence: Optional[int] = Field(None, ge=0, le=255)  # rpm
    fractional_cadence: Optional[float] = Field(None, ge=0, le=255)
    
    # Power
    power: Optional[int] = Field(None, ge=0, le=2000)  # watts
    left_right_balance: Optional[float] = Field(None)  # Can be scaled in FIT
    left_torque_effectiveness: Optional[float] = Field(None, ge=0, le=100)
    right_torque_effectiveness: Optional[float] = Field(None, ge=0, le=100)
    left_pedal_smoothness: Optional[float] = Field(None, ge=0, le=100)
    right_pedal_smoothness: Optional[float] = Field(None, ge=0, le=100)
    combined_pedal_smoothness: Optional[float] = Field(None, ge=0, le=100)
    left_pco: Optional[float] = None  # Platform center offset
    right_pco: Optional[float] = None
    left_power_phase: Optional[List[float]] = None
    right_power_phase: Optional[List[float]] = None
    left_power_phase_peak: Optional[List[float]] = None
    right_power_phase_peak: Optional[List[float]] = None
    
    # Temperature
    temperature: Optional[int] = Field(None, ge=-100, le=100)  # celsius
    
    # Grade
    grade: Optional[float] = Field(None, ge=-100, le=100)  # percent
    
    # Calories
    calories: Optional[int] = Field(None, ge=0)
    
    # GPS accuracy
    gps_accuracy: Optional[int] = Field(None, ge=0)  # meters
    
    # Vertical metrics
    vertical_oscillation: Optional[float] = Field(None, ge=0)  # mm
    stance_time_percent: Optional[float] = Field(None, ge=0, le=100)
    stance_time: Optional[float] = Field(None, ge=0)  # ms
    vertical_ratio: Optional[float] = Field(None, ge=0)
    stance_time_balance: Optional[float] = Field(None, ge=0, le=100)
    step_length: Optional[float] = Field(None, ge=0)  # mm
    
    # Device battery
    battery_soc: Optional[float] = Field(None, ge=0, le=100)  # State of charge %
    
    # Motor data (e-bike)
    motor_power: Optional[int] = Field(None, ge=0)  # watts
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        """Ensure timestamp is timezone aware."""
        if v and not v.tzinfo:
            from datetime import timezone
            return v.replace(tzinfo=timezone.utc)
        return v


class FITHRVData(BaseModel):
    """Heart Rate Variability data."""
    model_config = ConfigDict(extra='allow')
    
    time: Optional[List[float]] = None  # R-R intervals in ms


class FITEvent(BaseModel):
    """Event data from FIT file."""
    model_config = ConfigDict(extra='allow')
    
    timestamp: Optional[datetime] = None
    event: Optional[str] = None
    event_type: Optional[str] = None
    data: Optional[Any] = None
    event_group: Optional[int] = None


class FITZones(BaseModel):
    """Training zones configuration."""
    model_config = ConfigDict(extra='allow')
    
    max_heart_rate: Optional[int] = Field(None, ge=0, le=255)
    threshold_heart_rate: Optional[int] = Field(None, ge=0, le=255)
    functional_threshold_power: Optional[int] = Field(None, ge=0)
    
    hr_zones: Optional[List[int]] = None
    power_zones: Optional[List[int]] = None


class FITSummaryStats(BaseModel):
    """Calculated summary statistics from FIT records."""
    model_config = ConfigDict(extra='allow')
    
    # Time
    total_time: Optional[float] = Field(None, ge=0)
    recording_interval: Optional[float] = Field(None, ge=0)
    
    # Heart rate
    hr_avg: Optional[float] = Field(None, ge=0, le=255)
    hr_max: Optional[float] = Field(None, ge=0, le=255)
    hr_min: Optional[float] = Field(None, ge=0, le=255)
    hr_std: Optional[float] = Field(None, ge=0)
    hr_zones: Optional[Dict[str, float]] = None  # Percentage in each zone
    
    # Power
    power_avg: Optional[float] = Field(None, ge=0)
    power_max: Optional[float] = Field(None, ge=0)
    power_normalized: Optional[float] = Field(None, ge=0)
    power_variability_index: Optional[float] = Field(None, ge=0)
    power_zones: Optional[Dict[str, float]] = None  # Percentage in each zone
    power_left_right_balance: Optional[float] = Field(None, ge=0, le=100)
    
    # Cadence
    cadence_avg: Optional[float] = Field(None, ge=0)
    cadence_max: Optional[float] = Field(None, ge=0)
    
    # Speed
    speed_avg: Optional[float] = Field(None, ge=0)
    speed_max: Optional[float] = Field(None, ge=0)
    
    # Altitude
    altitude_min: Optional[float] = None
    altitude_max: Optional[float] = None
    altitude_gain: Optional[float] = Field(None, ge=0)
    
    # Temperature
    temp_avg: Optional[float] = None
    temp_min: Optional[float] = None
    temp_max: Optional[float] = None
    
    # Advanced metrics
    training_peaks_score: Optional[float] = Field(None, ge=0)
    intensity_factor: Optional[float] = Field(None, ge=0)
    efficiency_factor: Optional[float] = Field(None, ge=0)
    
    @model_validator(mode='after')
    def calculate_derived_metrics(self):
        """Calculate derived metrics."""
        # Calculate Intensity Factor if we have normalized and threshold power
        if self.power_normalized and hasattr(self, 'ftp'):
            self.intensity_factor = self.power_normalized / getattr(self, 'ftp', 250)
        
        # Calculate Efficiency Factor if we have normalized power and avg HR
        if self.power_normalized and self.hr_avg:
            self.efficiency_factor = self.power_normalized / self.hr_avg
        
        return self


class FITParseResult(BaseModel):
    """Complete result from parsing a FIT file."""
    model_config = ConfigDict(extra='allow')
    
    file_path: str
    session_data: Optional[FITSessionData] = None
    lap_data: List[FITLapData] = Field(default_factory=list)
    record_data: List[FITRecordData] = Field(default_factory=list)
    hrv_data: List[FITHRVData] = Field(default_factory=list)
    device_info: Optional[FITDeviceInfo] = None
    zones: Optional[FITZones] = None
    events: List[FITEvent] = Field(default_factory=list)
    summary_stats: Optional[FITSummaryStats] = None
    parsing_errors: List[str] = Field(default_factory=list)
    
    # DataFrame representation
    records_df: Optional[Any] = None  # Will be pl.DataFrame
    
    @field_validator('records_df')
    @classmethod
    def validate_dataframe(cls, v):
        """Ensure records_df is a Polars DataFrame."""
        if v is not None and not isinstance(v, pl.DataFrame):
            raise ValueError("records_df must be a Polars DataFrame")
        return v
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for database storage."""
        result = {}
        
        # Add session data fields with fit_ prefix
        if self.session_data:
            for key, value in self.session_data.model_dump(exclude_none=True).items():
                result[f"fit_{key}"] = value
        
        # Add summary stats with fit_ prefix
        if self.summary_stats:
            for key, value in self.summary_stats.model_dump(exclude_none=True).items():
                if isinstance(value, dict):
                    # Flatten zone distributions
                    for zone_key, zone_value in value.items():
                        result[f"fit_{key}_{zone_key}"] = zone_value
                else:
                    result[f"fit_{key}"] = value
        
        # Add device info
        if self.device_info:
            for key, value in self.device_info.model_dump(exclude_none=True).items():
                result[f"fit_device_{key}"] = value
        
        # Add metadata
        result["fit_data_points"] = len(self.record_data)
        result["fit_lap_count"] = len(self.lap_data)
        result["fit_event_count"] = len(self.events)
        result["has_fit_analysis"] = True
        
        return result
    
    def get_time_series_features(self) -> Dict[str, Any]:
        """Extract time-series features for ML models."""
        if not self.record_data:
            return {}
        
        features = {}
        
        # Power variability features
        if any(r.power for r in self.record_data):
            powers = [r.power for r in self.record_data if r.power is not None]
            if powers:
                features["power_std"] = np.std(powers)
                features["power_cv"] = np.std(powers) / np.mean(powers) if np.mean(powers) > 0 else 0
                features["power_q25"] = np.percentile(powers, 25)
                features["power_q50"] = np.percentile(powers, 50)
                features["power_q75"] = np.percentile(powers, 75)
                features["power_iqr"] = features["power_q75"] - features["power_q25"]
                features["power_skewness"] = self._calculate_skewness(powers)
                features["power_kurtosis"] = self._calculate_kurtosis(powers)
                
                # Power smoothness (lower is smoother)
                if len(powers) > 1:
                    power_diff = np.diff(powers)
                    features["power_smoothness"] = np.std(power_diff)
                
                # Time above threshold (% time above different power levels)
                for threshold in [100, 150, 200, 250, 300, 350, 400]:
                    features[f"time_above_{threshold}w_pct"] = (sum(1 for p in powers if p > threshold) / len(powers)) * 100
        
        # Heart rate variability features
        if any(r.heart_rate for r in self.record_data):
            hrs = [r.heart_rate for r in self.record_data if r.heart_rate is not None]
            if hrs:
                features["hr_std"] = np.std(hrs)
                features["hr_cv"] = np.std(hrs) / np.mean(hrs) if np.mean(hrs) > 0 else 0
                features["hr_drift"] = self._calculate_drift(hrs)
                features["hr_decoupling"] = self._calculate_decoupling(hrs, powers if 'powers' in locals() else None)
        
        # Cadence consistency
        if any(r.cadence for r in self.record_data):
            cadences = [r.cadence for r in self.record_data if r.cadence is not None and r.cadence > 0]
            if cadences:
                features["cadence_std"] = np.std(cadences)
                features["cadence_cv"] = np.std(cadences) / np.mean(cadences) if np.mean(cadences) > 0 else 0
        
        # Pedal balance features
        if any(r.left_right_balance for r in self.record_data):
            balances = [r.left_right_balance for r in self.record_data if r.left_right_balance is not None]
            if balances:
                features["pedal_balance_avg"] = np.mean(balances)
                features["pedal_balance_std"] = np.std(balances)
        
        # Torque effectiveness
        if any(r.left_torque_effectiveness for r in self.record_data):
            left_te = [r.left_torque_effectiveness for r in self.record_data if r.left_torque_effectiveness is not None]
            right_te = [r.right_torque_effectiveness for r in self.record_data if r.right_torque_effectiveness is not None]
            if left_te:
                features["torque_effectiveness_left_avg"] = np.mean(left_te)
            if right_te:
                features["torque_effectiveness_right_avg"] = np.mean(right_te)
        
        # Altitude changes
        if any(r.altitude for r in self.record_data):
            altitudes = [r.altitude for r in self.record_data if r.altitude is not None]
            if len(altitudes) > 1:
                features["altitude_change_rate"] = np.std(np.diff(altitudes))
        
        # Temperature adaptation
        if any(r.temperature for r in self.record_data):
            temps = [r.temperature for r in self.record_data if r.temperature is not None]
            if temps:
                features["temp_range"] = max(temps) - min(temps)
                features["temp_std"] = np.std(temps)
        
        return features
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return (n / ((n - 1) * (n - 2))) * np.sum(((np.array(data) - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 4:
            return 0
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((np.array(data) - mean) / std) ** 4) - \
               (3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    
    def _calculate_drift(self, data: List[float]) -> float:
        """Calculate drift (difference between first and second half averages)."""
        if len(data) < 2:
            return 0
        mid = len(data) // 2
        first_half = np.mean(data[:mid])
        second_half = np.mean(data[mid:])
        if first_half == 0:
            return 0
        return ((second_half - first_half) / first_half) * 100
    
    def _calculate_decoupling(self, hr_data: List[float], power_data: Optional[List[float]]) -> float:
        """Calculate aerobic decoupling (HR/Power drift difference)."""
        if not power_data or len(hr_data) != len(power_data):
            return 0
        
        hr_drift = self._calculate_drift(hr_data)
        power_drift = self._calculate_drift(power_data)
        
        return hr_drift - power_drift


class FITActivityEnhanced(BaseModel):
    """Activity record enhanced with FIT file data."""
    model_config = ConfigDict(extra='allow')
    
    # Base activity fields
    id: str
    name: str
    type: str
    start_date_local: datetime
    distance: Optional[float] = None
    moving_time: Optional[float] = None
    
    # All FIT-derived fields (flattened)
    fit_timestamp: Optional[datetime] = None
    fit_total_elapsed_time: Optional[float] = None
    fit_total_timer_time: Optional[float] = None
    fit_total_distance: Optional[float] = None
    fit_avg_power: Optional[float] = None
    fit_max_power: Optional[float] = None
    fit_normalized_power: Optional[float] = None
    fit_power_variability_index: Optional[float] = None
    fit_avg_heart_rate: Optional[float] = None
    fit_max_heart_rate: Optional[float] = None
    fit_hr_avg: Optional[float] = None
    fit_hr_max: Optional[float] = None
    fit_hr_std: Optional[float] = None
    fit_hr_zones_zone1: Optional[float] = None
    fit_hr_zones_zone2: Optional[float] = None
    fit_hr_zones_zone3: Optional[float] = None
    fit_hr_zones_zone4: Optional[float] = None
    fit_hr_zones_zone5: Optional[float] = None
    fit_power_zones_zone1: Optional[float] = None
    fit_power_zones_zone2: Optional[float] = None
    fit_power_zones_zone3: Optional[float] = None
    fit_power_zones_zone4: Optional[float] = None
    fit_power_zones_zone5: Optional[float] = None
    fit_power_zones_zone6: Optional[float] = None
    fit_power_zones_zone7: Optional[float] = None
    fit_cadence_avg: Optional[float] = None
    fit_cadence_max: Optional[float] = None
    fit_speed_avg: Optional[float] = None
    fit_speed_max: Optional[float] = None
    fit_altitude_gain: Optional[float] = None
    fit_temp_avg: Optional[float] = None
    fit_data_points: Optional[int] = None
    fit_lap_count: Optional[int] = None
    has_fit_analysis: bool = False
    fit_timeseries_file: Optional[str] = None
    
    # Time-series features for ML
    fit_time_series_features: Optional[Dict[str, float]] = None