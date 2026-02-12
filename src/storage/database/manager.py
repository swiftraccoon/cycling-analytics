"""SQLite database manager for cycling analytics."""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import polars as pl
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import DATABASE_PATH
from src.storage.database.schema_migrator import SchemaMigrator

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage SQLite database operations."""
    
    def __init__(self, db_path: str = None):
        """Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file (uses config default if None)
        """
        self.db_path = Path(db_path if db_path else DATABASE_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # SQLAlchemy engine with WAL mode for better concurrency
        self.engine = self._create_engine()
        
        # Initialize database
        self._initialize_database()
    
    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine with optimized settings."""
        engine = create_engine(
            f"sqlite:///{self.db_path}",
            connect_args={
                "check_same_thread": False,
                "timeout": 30,
            },
            pool_pre_ping=True,
            echo=False,
        )
        
        # Enable WAL mode for better concurrency
        with engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.execute(text("PRAGMA synchronous=NORMAL"))
            conn.execute(text("PRAGMA cache_size=10000"))
            conn.execute(text("PRAGMA temp_store=MEMORY"))
            conn.commit()
        
        return engine
    
    def _initialize_database(self):
        """Initialize database schema."""
        with self.engine.connect() as conn:
            # Create activities table with all columns
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS activities (
                    -- Primary key
                    id TEXT PRIMARY KEY,
                    
                    -- Temporal fields
                    start_date_local TIMESTAMP,
                    start_date TIMESTAMP,
                    icu_sync_date TIMESTAMP,
                    
                    -- Basic info
                    name TEXT,
                    type TEXT,
                    gear TEXT,
                    description TEXT,
                    
                    -- Duration and distance
                    moving_time REAL,
                    distance REAL,
                    elapsed_time REAL,
                    total_elevation_gain REAL,
                    icu_recording_time REAL,
                    icu_warmup_time REAL,
                    icu_cooldown_time REAL,
                    
                    -- Speed metrics
                    max_speed REAL,
                    average_speed REAL,
                    pace REAL,
                    threshold_pace REAL,
                    
                    -- Heart rate
                    has_heartrate BOOLEAN,
                    max_heartrate INTEGER,
                    average_heartrate INTEGER,
                    lthr INTEGER,
                    icu_resting_hr INTEGER,
                    icu_hrrc INTEGER,
                    icu_hrrc_start_bpm INTEGER,
                    
                    -- Power data
                    device_watts BOOLEAN,
                    icu_average_watts REAL,
                    icu_normalized_watts REAL,
                    icu_joules REAL,
                    icu_intensity REAL,
                    icu_ftp INTEGER,
                    icu_eftp REAL,
                    icu_w_prime REAL,
                    icu_power_spike_threshold REAL,
                    icu_pm_ftp INTEGER,
                    icu_pm_cp INTEGER,
                    icu_pm_w_prime REAL,
                    icu_pm_p_max INTEGER,
                    
                    -- Training metrics
                    icu_training_load REAL,
                    icu_training_load_edited REAL,
                    icu_fatigue REAL,
                    icu_fitness REAL,
                    icu_variability REAL,
                    icu_efficiency REAL,
                    icu_rpe REAL,
                    power_load REAL,
                    hr_load REAL,
                    pace_load REAL,
                    
                    -- Cadence
                    average_cadence REAL,
                    
                    -- Other metrics
                    calories INTEGER,
                    icu_weight REAL,
                    compliance REAL,
                    
                    -- Flags
                    trainer BOOLEAN,
                    commute BOOLEAN,
                    race BOOLEAN,
                    icu_ignore_power BOOLEAN,
                    icu_ignore_hr BOOLEAN,
                    icu_ignore_time BOOLEAN,
                    
                    -- Heart rate zones (seconds)
                    hr_z1 INTEGER,
                    hr_z2 INTEGER,
                    hr_z3 INTEGER,
                    hr_z4 INTEGER,
                    hr_z5 INTEGER,
                    hr_z6 INTEGER,
                    hr_max INTEGER,
                    hr_z1_secs INTEGER,
                    hr_z2_secs INTEGER,
                    hr_z3_secs INTEGER,
                    hr_z4_secs INTEGER,
                    hr_z5_secs INTEGER,
                    hr_z6_secs INTEGER,
                    hr_z7_secs INTEGER,
                    
                    -- Power zones (seconds)
                    z1_secs INTEGER,
                    z2_secs INTEGER,
                    z3_secs INTEGER,
                    z4_secs INTEGER,
                    z5_secs INTEGER,
                    z6_secs INTEGER,
                    z7_secs INTEGER,
                    sweet_spot_secs INTEGER,
                    
                    -- Metadata
                    timezone TEXT,
                    file_type TEXT,
                    external_id TEXT,
                    
                    -- Ingestion metadata
                    file_source TEXT,
                    file_hash TEXT,
                    import_timestamp TIMESTAMP,
                    all_file_sources TEXT,
                    
                    -- Deduplication flags
                    is_exact_duplicate BOOLEAN,
                    is_potential_duplicate BOOLEAN,
                    duplicate_count INTEGER,
                    
                    -- Timestamps
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Create indexes for common queries (only if columns exist)
            # Get existing columns
            result = conn.execute(text("PRAGMA table_info(activities)"))
            columns = [row[1] for row in result]
            
            # Only create indexes for existing columns
            if 'start_date_local' in columns:
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_start_date ON activities(start_date_local)"))
            if 'type' in columns:
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_type ON activities(type)"))
            if 'file_hash' in columns:
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_file_hash ON activities(file_hash)"))
            if 'import_timestamp' in columns:
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_import_timestamp ON activities(import_timestamp)"))
            
            # Create ingestion history table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_hash TEXT NOT NULL UNIQUE,
                    file_size INTEGER,
                    record_count INTEGER,
                    new_activities INTEGER,
                    duplicate_activities INTEGER,
                    updated_activities INTEGER,
                    ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT,
                    error_message TEXT
                )
            """))
            
            # Create data quality metrics table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS data_quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    check_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_records INTEGER,
                    valid_records INTEGER,
                    invalid_records INTEGER,
                    completeness_score REAL,
                    consistency_score REAL,
                    validity_score REAL,
                    overall_score REAL,
                    details JSON
                )
            """))
            
            conn.commit()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def save_activities(self, df: pl.DataFrame, update_existing: bool = True) -> dict:
        """Save activities to database with strict validation.
        
        Args:
            df: DataFrame with activity data
            update_existing: Whether to update existing records
            
        Returns:
            Dictionary with save statistics
        """
        if df.is_empty():
            return {"saved": 0, "updated": 0, "skipped": 0}
        
        # CRITICAL: Validate and normalize all data before saving
        from src.data.validator import DataValidator
        try:
            df = DataValidator.validate_dataframe(df)
            logger.info(f"Validated {len(df)} activities")
            
            # Check data quality - MUST be high quality
            quality_metrics = DataValidator.ensure_data_quality(df)
            logger.info(f"Data quality score: {quality_metrics['quality_score']:.1f}%")
            
            # STRICT: Require at least 95% data quality for core fields
            if quality_metrics['quality_score'] < 95:
                logger.error(f"UNACCEPTABLE data quality: {quality_metrics['quality_score']:.1f}%")
                logger.error("Data quality MUST be at least 95%! Required fields with issues:")
                for col, pct in quality_metrics['null_percentages'].items():
                    if pct > 10:  # More than 10% null is unacceptable for core fields
                        logger.error(f"  {col}: {pct:.1f}% null - MUST BE FIXED")
                
                # For critical fields, NO nulls allowed
                critical_fields = ['id', 'start_date_local', 'name', 'type']
                for field in critical_fields:
                    if field in quality_metrics['null_percentages']:
                        if quality_metrics['null_percentages'][field] > 0:
                            raise ValueError(f"CRITICAL FIELD '{field}' has null values - THIS IS NOT ACCEPTABLE")
                
                # Fail if quality is too low - NO COMPROMISES ON DATA QUALITY
                if quality_metrics['quality_score'] < 95:
                    raise ValueError(f"Data quality score {quality_metrics['quality_score']:.1f}% is UNACCEPTABLE. Minimum is 95%!")
        except ValueError as e:
            error_msg = str(e)
            logger.error(f"Data validation failed: {e}")
            # Re-raise quality enforcement errors so callers can catch them
            if "UNACCEPTABLE" in error_msg or "CRITICAL FIELD" in error_msg or "quality score" in error_msg.lower():
                raise ValueError(f"Cannot save invalid data: {e}")
            return {"saved": 0, "updated": 0, "skipped": 0, "errors": error_msg}
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return {"saved": 0, "updated": 0, "skipped": 0, "errors": str(e)}
        
        # First, migrate schema to add any new columns
        migrator = SchemaMigrator(self.engine)
        migration_result = migrator.add_missing_columns("activities", df)
        if migration_result["count"] > 0:
            logger.info(f"Added {migration_result['count']} new columns to database schema")
        
        # Extract zone data from JSON before saving
        df = self._extract_zone_data(df)
        
        # Now all columns should be valid
        df_columns = df.columns
        df = df.select(df_columns)
        
        # Convert Polars to pandas for SQLAlchemy compatibility
        df_pandas = df.to_pandas()
        
        # Add updated_at timestamp
        df_pandas["updated_at"] = datetime.now()
        
        # Get existing IDs
        existing_ids = set()
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT id FROM activities"))
            existing_ids = {row[0] for row in result}
        
        # Separate new and existing records
        new_records = df_pandas[~df_pandas["id"].isin(existing_ids)]
        existing_records = df_pandas[df_pandas["id"].isin(existing_ids)]
        
        stats = {"saved": 0, "updated": 0, "skipped": 0}
        
        # Save new records
        if not new_records.empty:
            new_records.to_sql(
                "activities",
                self.engine,
                if_exists="append",
                index=False,
                method="multi",
            )
            stats["saved"] = len(new_records)
            logger.info(f"Saved {len(new_records)} new activities")
        
        # Update existing records if requested
        if update_existing and not existing_records.empty:
            with self.engine.connect() as conn:
                for _, row in existing_records.iterrows():
                    update_query = text("""
                        UPDATE activities 
                        SET updated_at = :updated_at,
                            file_source = :file_source,
                            all_file_sources = :all_file_sources
                        WHERE id = :id
                    """)
                    conn.execute(update_query, {
                        "id": row["id"],
                        "updated_at": datetime.now(),
                        "file_source": row.get("file_source"),
                        "all_file_sources": row.get("all_file_sources"),
                    })
                conn.commit()
            stats["updated"] = len(existing_records)
            logger.info(f"Updated {len(existing_records)} existing activities")
        else:
            stats["skipped"] = len(existing_records)
        
        return stats
    
    def get_activities(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        activity_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pl.DataFrame:
        """Retrieve activities from database.
        
        Args:
            start_date: Filter by start date
            end_date: Filter by end date
            activity_type: Filter by activity type
            limit: Maximum number of records
            
        Returns:
            DataFrame with activities
        """
        query = "SELECT * FROM activities WHERE 1=1"
        params = {}
        
        if start_date:
            query += " AND start_date_local >= :start_date"
            # Ensure datetime is formatted properly for SQLite
            if isinstance(start_date, datetime):
                params["start_date"] = start_date.strftime("%Y-%m-%d %H:%M:%S")
            else:
                params["start_date"] = start_date
        
        if end_date:
            query += " AND start_date_local <= :end_date"
            # For end date, we want to include the entire day
            if isinstance(end_date, datetime):
                # If it's midnight, extend to end of day
                if end_date.hour == 0 and end_date.minute == 0 and end_date.second == 0:
                    end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                params["end_date"] = end_date.strftime("%Y-%m-%d %H:%M:%S.%f")
            else:
                params["end_date"] = end_date
        
        if activity_type:
            query += " AND type = :activity_type"
            params["activity_type"] = activity_type
        
        query += " ORDER BY start_date_local DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        # Read from database using pandas intermediary
        import pandas as pd
        
        # Log the query for debugging
        logger.debug(f"Query: {query}")
        logger.debug(f"Params: {params}")
        
        df_pandas = pd.read_sql(query, self.engine, params=params)
        
        # Convert date columns to datetime if they're strings
        date_cols = ['start_date_local', 'start_date', 'icu_sync_date']
        for col in date_cols:
            if col in df_pandas.columns and df_pandas[col].dtype == 'object':
                df_pandas[col] = pd.to_datetime(df_pandas[col], errors='coerce')
        
        # Convert numeric columns that might be stored as strings
        numeric_cols = [
            'moving_time', 'distance', 'elapsed_time', 'total_elevation_gain',
            'icu_recording_time', 'icu_warmup_time', 'icu_cooldown_time',
            'max_speed', 'average_speed', 'pace', 'threshold_pace',
            'icu_average_watts', 'icu_normalized_watts', 'icu_joules', 'icu_intensity',
            'icu_ftp', 'icu_eftp', 'icu_w_prime', 'icu_power_spike_threshold',
            'icu_pm_ftp', 'icu_pm_cp', 'icu_pm_w_prime', 'icu_pm_p_max',
            'icu_training_load', 'icu_training_load_edited', 'icu_fatigue', 'icu_fitness',
            'icu_variability', 'icu_efficiency', 'icu_rpe', 'power_load', 'hr_load',
            'pace_load', 'average_cadence', 'icu_weight', 'compliance',
            'normalized_power', 'intensity_factor', 'training_stress_score',
            'average_power', 'max_power', 'duration', 'moving_duration', 'elapsed_duration',
            'elevation_gain', 'elevation_loss', 'max_temperature', 'min_temperature',
            'vo2_max', 'lactate_threshold_hr', 'avg_ground_contact_time',
            'avg_stride_length', 'avg_vertical_oscillation', 'avg_vertical_ratio',
            'avg_ground_contact_balance', 'max_cadence'
        ]
        
        integer_cols = [
            'max_heartrate', 'average_heartrate', 'lthr', 'icu_resting_hr',
            'icu_hrrc', 'icu_hrrc_start_bpm', 'calories', 'pr_count',
            'hr_z1', 'hr_z2', 'hr_z3', 'hr_z4', 'hr_z5', 'hr_z6', 'hr_max',
            'hr_z1_secs', 'hr_z2_secs', 'hr_z3_secs', 'hr_z4_secs', 'hr_z5_secs',
            'hr_z6_secs', 'hr_z7_secs', 'z1_secs', 'z2_secs', 'z3_secs', 'z4_secs',
            'z5_secs', 'z6_secs', 'z7_secs', 'sweet_spot_secs', 'average_hr', 'max_hr',
            'bmr_calories'
        ]
        
        # Convert numeric columns
        for col in numeric_cols:
            if col in df_pandas.columns and df_pandas[col].dtype == 'object':
                df_pandas[col] = pd.to_numeric(df_pandas[col], errors='coerce')
        
        for col in integer_cols:
            if col in df_pandas.columns and df_pandas[col].dtype == 'object':
                df_pandas[col] = pd.to_numeric(df_pandas[col], errors='coerce')
        
        df = pl.from_pandas(df_pandas)
        
        logger.info(f"Retrieved {len(df)} activities from database")
        return df
    
    def save_ingestion_record(self, metadata: dict, status: str = "success", error: str = None):
        """Save ingestion history record.
        
        Args:
            metadata: File metadata dictionary
            status: Ingestion status
            error: Error message if failed
        """
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT OR REPLACE INTO ingestion_history 
                (file_path, file_name, file_hash, file_size, record_count,
                 new_activities, duplicate_activities, updated_activities,
                 status, error_message)
                VALUES (:file_path, :file_name, :file_hash, :file_size, :record_count,
                        :new_activities, :duplicate_activities, :updated_activities,
                        :status, :error_message)
            """), {
                "file_path": metadata.get("file_path"),
                "file_name": metadata.get("file_name"),
                "file_hash": metadata.get("file_hash"),
                "file_size": metadata.get("file_size"),
                "record_count": metadata.get("record_count", 0),
                "new_activities": metadata.get("new_activities", 0),
                "duplicate_activities": metadata.get("duplicate_activities", 0),
                "updated_activities": metadata.get("updated_activities", 0),
                "status": status,
                "error_message": error,
            })
            conn.commit()
    
    def _extract_zone_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract zone data from JSON fields.
        
        Args:
            df: DataFrame with activities
            
        Returns:
            DataFrame with extracted zone data
        """
        import json
        
        # Convert to pandas for easier manipulation
        df_pandas = df.to_pandas()
        
        # Extract HR zone data if present
        if 'hr_zones_data' in df_pandas.columns:
            for idx, row in df_pandas.iterrows():
                hr_zones_json = row.get('hr_zones_data')
                if hr_zones_json:
                    try:
                        # Parse JSON string
                        if isinstance(hr_zones_json, str):
                            zones_data = json.loads(hr_zones_json)
                        else:
                            zones_data = hr_zones_json
                        
                        # Extract zone times
                        if isinstance(zones_data, list):
                            for zone_info in zones_data:
                                if isinstance(zone_info, dict):
                                    zone_num = zone_info.get('zoneNumber')
                                    secs_in_zone = zone_info.get('secsInZone', 0)
                                    
                                    if zone_num and 1 <= zone_num <= 7:
                                        col_name = f'hr_z{zone_num}_secs'
                                        if col_name not in df_pandas.columns:
                                            df_pandas[col_name] = 0
                                        df_pandas.at[idx, col_name] = secs_in_zone
                    except (json.JSONDecodeError, TypeError):
                        pass
        
        # Extract power zone data if present (from FIT files)
        if 'power_zones_data' in df_pandas.columns:
            for idx, row in df_pandas.iterrows():
                power_zones_json = row.get('power_zones_data')
                if power_zones_json:
                    try:
                        if isinstance(power_zones_json, str):
                            zones_data = json.loads(power_zones_json)
                        else:
                            zones_data = power_zones_json
                        
                        if isinstance(zones_data, list):
                            for zone_info in zones_data:
                                if isinstance(zone_info, dict):
                                    zone_num = zone_info.get('zoneNumber')
                                    secs_in_zone = zone_info.get('secsInZone', 0)
                                    
                                    if zone_num and 1 <= zone_num <= 7:
                                        col_name = f'z{zone_num}_secs'
                                        if col_name not in df_pandas.columns:
                                            df_pandas[col_name] = 0
                                        df_pandas.at[idx, col_name] = secs_in_zone
                    except (json.JSONDecodeError, TypeError):
                        pass
        
        # Convert back to polars
        return pl.from_pandas(df_pandas)
    
    def get_ingestion_history(self, limit: int = 10) -> pl.DataFrame:
        """Get recent ingestion history.
        
        Args:
            limit: Number of records to retrieve
            
        Returns:
            DataFrame with ingestion history
        """
        query = """
            SELECT * FROM ingestion_history 
            ORDER BY ingestion_timestamp DESC 
            LIMIT :limit
        """
        
        import pandas as pd
        df_pandas = pd.read_sql(query, self.engine, params={"limit": limit})
        df = pl.from_pandas(df_pandas)
        return df
    
    def get_summary_stats(self) -> dict:
        """Get summary statistics from database.
        
        Returns:
            Dictionary with summary statistics
        """
        with self.engine.connect() as conn:
            stats = {}
            
            # Total activities
            result = conn.execute(text("SELECT COUNT(*) FROM activities"))
            stats["total_activities"] = result.scalar()
            
            # Date range
            result = conn.execute(text("""
                SELECT MIN(start_date_local), MAX(start_date_local) 
                FROM activities
            """))
            row = result.first()
            if row:
                stats["earliest_activity"] = row[0]
                stats["latest_activity"] = row[1]
            
            # Activities by type
            result = conn.execute(text("""
                SELECT type, COUNT(*) 
                FROM activities 
                GROUP BY type
            """))
            stats["activities_by_type"] = dict(result.fetchall())
            
            # Total distance and time
            result = conn.execute(text("""
                SELECT SUM(distance), SUM(moving_time) 
                FROM activities
            """))
            row = result.first()
            if row:
                stats["total_distance"] = row[0]
                stats["total_moving_time"] = row[1]
            
            # Files processed
            result = conn.execute(text("SELECT COUNT(*) FROM ingestion_history"))
            stats["files_processed"] = result.scalar()
        
        return stats