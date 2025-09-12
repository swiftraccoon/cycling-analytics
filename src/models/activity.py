"""Activity data models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Activity(BaseModel):
    """Model for cycling activity data."""
    
    # Core identifiers
    id: str
    start_date_local: datetime
    name: str
    type: str
    
    # Duration and distance
    moving_time: Optional[float] = None
    distance: Optional[float] = None
    elapsed_time: Optional[float] = None
    total_elevation_gain: Optional[float] = None
    
    # Speed metrics
    max_speed: Optional[float] = None
    average_speed: Optional[float] = None
    pace: Optional[float] = None
    
    # Heart rate data
    has_heartrate: bool = False
    max_heartrate: Optional[int] = None
    average_heartrate: Optional[int] = None
    lthr: Optional[int] = None
    
    # Power data
    device_watts: Optional[bool] = None
    icu_average_watts: Optional[float] = None
    icu_normalized_watts: Optional[float] = None
    icu_joules: Optional[float] = None
    icu_intensity: Optional[float] = None
    icu_ftp: Optional[int] = None
    icu_eftp: Optional[float] = None
    
    # Training metrics
    icu_training_load: Optional[float] = None
    icu_fatigue: Optional[float] = None
    icu_fitness: Optional[float] = None
    icu_variability: Optional[float] = None
    icu_efficiency: Optional[float] = None
    
    # Cadence
    average_cadence: Optional[float] = None
    
    # Other metrics
    calories: Optional[int] = None
    icu_weight: Optional[float] = None
    
    # Flags
    trainer: bool = False
    commute: bool = False
    race: bool = False
    
    # Zone time in seconds
    hr_z1_secs: Optional[int] = None
    hr_z2_secs: Optional[int] = None
    hr_z3_secs: Optional[int] = None
    hr_z4_secs: Optional[int] = None
    hr_z5_secs: Optional[int] = None
    hr_z6_secs: Optional[int] = None
    hr_z7_secs: Optional[int] = None
    
    z1_secs: Optional[int] = None
    z2_secs: Optional[int] = None
    z3_secs: Optional[int] = None
    z4_secs: Optional[int] = None
    z5_secs: Optional[int] = None
    z6_secs: Optional[int] = None
    z7_secs: Optional[int] = None
    
    # Metadata
    start_date: Optional[datetime] = None
    icu_sync_date: Optional[datetime] = None
    timezone: Optional[str] = None
    file_type: Optional[str] = None
    external_id: Optional[str] = None
    gear: Optional[str] = None
    description: Optional[str] = None
    
    # Ingestion metadata (added by our system)
    file_source: Optional[str] = None
    file_hash: Optional[str] = None
    import_timestamp: Optional[datetime] = None
    
    class Config:
        """Pydantic config."""
        str_strip_whitespace = True


class IngestionReport(BaseModel):
    """Model for file ingestion report."""
    
    file_path: str
    file_hash: str
    import_timestamp: datetime
    total_records: int
    new_activities: int
    duplicate_activities: int
    updated_activities: int
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    data_quality_score: float = 0.0


class ValidationReport(BaseModel):
    """Model for data validation report."""
    
    total_records: int
    valid_records: int
    invalid_records: int
    validation_errors: list[dict] = Field(default_factory=list)
    warnings: list[dict] = Field(default_factory=list)
    data_quality_metrics: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)