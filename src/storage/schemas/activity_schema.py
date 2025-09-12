"""
Activity database schema for cycling analytics data.

This module defines the SQLite schema for storing activity data with all 81 columns
from the original CSV files plus metadata columns for data lineage tracking.
"""

import sqlite3
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Complete activity table schema with all 81 columns from intervals.icu export
ACTIVITY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS activities (
    -- Primary key and basic identifiers
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_id TEXT UNIQUE NOT NULL,  -- Original activity ID from CSV
    
    -- Basic activity information
    name TEXT,
    type TEXT,
    start_date_local TEXT,
    distance REAL,
    moving_time INTEGER,
    elapsed_time INTEGER,
    total_elevation_gain REAL,
    
    -- Power metrics
    weighted_average_power REAL,
    power_meter BOOLEAN,
    device_watts BOOLEAN,
    max_watts REAL,
    normalized_power REAL,
    left_right_balance REAL,
    left_torque_effectiveness REAL,
    left_pedal_smoothness REAL,
    right_torque_effectiveness REAL,
    right_pedal_smoothness REAL,
    left_platform_center_offset REAL,
    right_platform_center_offset REAL,
    left_power_phase_start_angle REAL,
    left_power_phase_end_angle REAL,
    left_power_phase_angle REAL,
    left_peak_power_phase_start_angle REAL,
    left_peak_power_phase_end_angle REAL,
    left_peak_power_phase_angle REAL,
    right_power_phase_start_angle REAL,
    right_power_phase_end_angle REAL,
    right_power_phase_angle REAL,
    right_peak_power_phase_start_angle REAL,
    right_peak_power_phase_end_angle REAL,
    right_peak_power_phase_angle REAL,
    
    -- Heart rate metrics
    has_heartrate BOOLEAN,
    average_heartrate REAL,
    max_heartrate REAL,
    
    -- Speed and cadence metrics
    average_speed REAL,
    max_speed REAL,
    average_cadence REAL,
    max_cadence REAL,
    average_temp REAL,
    
    -- Training metrics
    kilojoules REAL,
    average_watts REAL,
    intensity_factor REAL,
    training_stress_score REAL,
    
    -- Location and weather
    start_latlng TEXT,
    end_latlng TEXT,
    location_city TEXT,
    location_state TEXT,
    location_country TEXT,
    timezone TEXT,
    utc_offset REAL,
    
    -- Gear and equipment
    gear_id TEXT,
    
    -- Strava specific fields
    external_id TEXT,
    upload_id TEXT,
    upload_id_str TEXT,
    
    -- Activity flags and status
    manual BOOLEAN,
    private BOOLEAN,
    flagged BOOLEAN,
    workout_type INTEGER,
    commute BOOLEAN,
    trainer BOOLEAN,
    description TEXT,
    
    -- Environmental conditions
    perceived_exertion REAL,
    
    -- Additional metrics
    suffer_score REAL,
    calories REAL,
    
    -- Lap information
    average_grade REAL,
    max_grade REAL,
    elev_high REAL,
    elev_low REAL,
    
    -- Detailed power metrics
    relative_effort REAL,
    
    -- Advanced metrics (intervals.icu specific)
    efficiency_factor REAL,
    variability_index REAL,
    
    -- Additional intervals.icu fields
    power_load REAL,
    hrss REAL,
    trimp REAL,
    
    -- Metadata columns for data lineage
    file_source TEXT NOT NULL,           -- Source file path
    import_timestamp TEXT NOT NULL,      -- When this record was imported
    file_hash TEXT NOT NULL,            -- SHA-256 hash of source file
    record_hash TEXT NOT NULL,          -- SHA-256 hash of this specific record
    is_duplicate BOOLEAN DEFAULT FALSE, -- Flag for duplicate detection
    original_row_number INTEGER,        -- Original row in source file
    
    -- Indexes
    UNIQUE(activity_id, file_hash)      -- Prevent duplicates from same file
);
"""

# Indexes for performance optimization
ACTIVITY_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_activity_id ON activities(activity_id);",
    "CREATE INDEX IF NOT EXISTS idx_start_date ON activities(start_date_local);",
    "CREATE INDEX IF NOT EXISTS idx_type ON activities(type);",
    "CREATE INDEX IF NOT EXISTS idx_file_source ON activities(file_source);",
    "CREATE INDEX IF NOT EXISTS idx_import_timestamp ON activities(import_timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_file_hash ON activities(file_hash);",
    "CREATE INDEX IF NOT EXISTS idx_is_duplicate ON activities(is_duplicate);",
]

# File ingestion tracking table
INGESTION_HISTORY_SQL = """
CREATE TABLE IF NOT EXISTS ingestion_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    file_name TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    file_hash TEXT UNIQUE NOT NULL,
    import_timestamp TEXT NOT NULL,
    records_count INTEGER NOT NULL,
    duplicates_found INTEGER DEFAULT 0,
    validation_status TEXT NOT NULL,  -- 'success', 'warning', 'error'
    validation_errors TEXT,           -- JSON string of validation issues
    processing_time_ms INTEGER,
    moved_to_archive BOOLEAN DEFAULT FALSE,
    archive_timestamp TEXT,
    
    UNIQUE(file_path, file_hash)
);
"""

INGESTION_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_ingestion_file_hash ON ingestion_history(file_hash);",
    "CREATE INDEX IF NOT EXISTS idx_ingestion_timestamp ON ingestion_history(import_timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_validation_status ON ingestion_history(validation_status);",
]

# Data quality metrics table
DATA_QUALITY_SQL = """
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    metric_description TEXT,
    severity TEXT,  -- 'info', 'warning', 'error'
    created_timestamp TEXT NOT NULL,
    
    FOREIGN KEY (file_hash) REFERENCES ingestion_history(file_hash),
    UNIQUE(file_hash, metric_name)
);
"""

def configure_database(db_path: Path) -> None:
    """
    Configure SQLite database with WAL mode and performance optimizations.
    
    Args:
        db_path: Path to the SQLite database file
    """
    try:
        with sqlite3.connect(str(db_path)) as conn:
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL;")
            
            # Performance optimizations
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA cache_size=10000;")
            conn.execute("PRAGMA temp_store=memory;")
            conn.execute("PRAGMA mmap_size=268435456;")  # 256MB
            
            # Foreign key support
            conn.execute("PRAGMA foreign_keys=ON;")
            
            conn.commit()
            logger.info(f"Database configured at {db_path}")
            
    except sqlite3.Error as e:
        logger.error(f"Failed to configure database: {e}")
        raise


def create_schema(db_path: Path) -> None:
    """
    Create the complete database schema with all tables and indexes.
    
    Args:
        db_path: Path to the SQLite database file
    """
    try:
        with sqlite3.connect(str(db_path)) as conn:
            # Create main tables
            conn.execute(ACTIVITY_TABLE_SQL)
            conn.execute(INGESTION_HISTORY_SQL)
            conn.execute(DATA_QUALITY_SQL)
            
            # Create indexes
            for index_sql in ACTIVITY_INDEXES_SQL:
                conn.execute(index_sql)
            
            for index_sql in INGESTION_INDEXES_SQL:
                conn.execute(index_sql)
            
            conn.commit()
            logger.info("Database schema created successfully")
            
    except sqlite3.Error as e:
        logger.error(f"Failed to create database schema: {e}")
        raise


def get_schema_version(db_path: Path) -> Optional[str]:
    """
    Get the current schema version from the database.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        Schema version string or None if not found
    """
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute("PRAGMA user_version;")
            version = cursor.fetchone()[0]
            return str(version) if version else None
            
    except sqlite3.Error as e:
        logger.error(f"Failed to get schema version: {e}")
        return None


def set_schema_version(db_path: Path, version: str) -> None:
    """
    Set the schema version in the database.
    
    Args:
        db_path: Path to the SQLite database file
        version: Version string to set
    """
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(f"PRAGMA user_version={version};")
            conn.commit()
            logger.info(f"Schema version set to {version}")
            
    except sqlite3.Error as e:
        logger.error(f"Failed to set schema version: {e}")
        raise


def initialize_database(db_path: Path, force_recreate: bool = False) -> None:
    """
    Initialize the database with schema and configuration.
    
    Args:
        db_path: Path to the SQLite database file
        force_recreate: Whether to drop and recreate existing tables
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    if force_recreate and db_path.exists():
        db_path.unlink()
        logger.info("Existing database removed for recreation")
    
    configure_database(db_path)
    create_schema(db_path)
    set_schema_version(db_path, "1")
    
    logger.info(f"Database initialized at {db_path}")