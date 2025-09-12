"""Database schema migrator to dynamically add new columns."""

import logging
from typing import Set, Dict, Any
from sqlalchemy import text
from sqlalchemy.engine import Engine
import polars as pl

logger = logging.getLogger(__name__)


class SchemaMigrator:
    """Handles dynamic schema updates for the database."""
    
    # Map Polars types to SQLite types
    TYPE_MAPPING = {
        pl.Float32: "REAL",
        pl.Float64: "REAL",
        pl.Int8: "INTEGER",
        pl.Int16: "INTEGER",
        pl.Int32: "INTEGER",
        pl.Int64: "INTEGER",
        pl.UInt8: "INTEGER",
        pl.UInt16: "INTEGER",
        pl.UInt32: "INTEGER",
        pl.UInt64: "INTEGER",
        pl.Boolean: "BOOLEAN",
        pl.Utf8: "TEXT",
        pl.Datetime: "TIMESTAMP",
        pl.Date: "DATE",
        pl.Time: "TIME",
    }
    
    @staticmethod
    def _determine_column_type(col_name: str) -> str:
        """Determine SQL type based on column name patterns.
        
        Args:
            col_name: Column name
            
        Returns:
            SQL type string
        """
        col = col_name.lower()
        
        # Numeric columns (REAL)
        numeric_patterns = [
            'moving_time', 'distance', 'elapsed_time', 'elevation_gain', 'elevation_loss',
            'speed', 'pace', 'watts', 'power', 'joules', 'intensity', 'load',
            'fitness', 'fatigue', 'variability', 'efficiency', 'cadence',
            'weight', 'compliance', 'temperature', 'oscillation', 'stride',
            'contact_time', 'contact_balance', 'vertical_ratio', 'duration',
            'normalized', 'factor', 'score', 'vo2', 'threshold'
        ]
        
        # Integer columns
        integer_patterns = [
            'heartrate', 'hr', 'ftp', 'lthr', 'bpm', 'calories', 'count',
            'secs', 'seconds', 'lap_count', 'event_count', 'data_points'
        ]
        
        # Boolean columns
        boolean_patterns = [
            'has_', 'is_', 'trainer', 'commute', 'race', 'ignore', 'device_watts'
        ]
        
        # Text columns
        text_patterns = [
            'name', 'type', 'gear', 'description', 'timezone', 'file',
            'path', 'source', 'hash', 'id', 'external', 'polyline', 'splits_data',
            'hr_zones_data', 'device_name', 'sport_type'
        ]
        
        # Timestamp columns
        timestamp_patterns = [
            'date', 'timestamp', 'created_at', 'updated_at', 'import_timestamp'
        ]
        
        # Check patterns
        for pattern in numeric_patterns:
            if pattern in col:
                return "REAL"
                
        for pattern in integer_patterns:
            if pattern in col:
                return "INTEGER"
                
        for pattern in boolean_patterns:
            if col.startswith(pattern):
                return "BOOLEAN"
                
        for pattern in timestamp_patterns:
            if pattern in col:
                return "TIMESTAMP"
                
        for pattern in text_patterns:
            if pattern in col:
                return "TEXT"
        
        # FIT-specific patterns
        if col.startswith('fit_'):
            if any(x in col for x in ['avg', 'max', 'min', 'std', 'normalized', 'vi', 'gain']):
                return "REAL"
            elif 'count' in col or 'points' in col:
                return "INTEGER"
                
        # Zone columns
        if col.startswith('z') and '_secs' in col:
            return "INTEGER"
        if col.startswith('hr_z') and not '_secs' in col:
            return "INTEGER"
            
        # Default to TEXT
        return "TEXT"
    
    def __init__(self, engine: Engine):
        """Initialize migrator with database engine.
        
        Args:
            engine: SQLAlchemy engine
        """
        self.engine = engine
    
    def get_existing_columns(self, table_name: str) -> Set[str]:
        """Get existing columns in a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Set of column names
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(f"PRAGMA table_info({table_name})"))
            return {row[1] for row in result}
    
    def add_missing_columns(self, table_name: str, df: pl.DataFrame) -> Dict[str, Any]:
        """Add any missing columns from DataFrame to table.
        
        Args:
            table_name: Name of the table
            df: DataFrame with potentially new columns
            
        Returns:
            Dictionary with migration statistics
        """
        existing_columns = self.get_existing_columns(table_name)
        df_columns = set(df.columns)
        
        # Find missing columns
        missing_columns = df_columns - existing_columns
        
        if not missing_columns:
            return {"added_columns": [], "count": 0}
        
        logger.info(f"Found {len(missing_columns)} new columns to add to {table_name}")
        
        added_columns = []
        
        with self.engine.connect() as conn:
            for col in missing_columns:
                # Determine SQL type based on column name patterns
                sql_type = self._determine_column_type(col)
                
                # Only fall back to dtype if we couldn't determine from name
                if sql_type == "TEXT":
                    pl_dtype = df[col].dtype
                    for pl_type, sql_type_str in self.TYPE_MAPPING.items():
                        if isinstance(pl_dtype, type(pl_type)):
                            sql_type = sql_type_str
                            break
                
                # Add the column
                try:
                    alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {col} {sql_type}"
                    conn.execute(text(alter_sql))
                    conn.commit()
                    added_columns.append(col)
                    logger.info(f"Added column: {col} ({sql_type})")
                except Exception as e:
                    logger.error(f"Failed to add column {col}: {e}")
        
        return {
            "added_columns": added_columns,
            "count": len(added_columns)
        }
    
    def migrate_fit_columns(self) -> Dict[str, Any]:
        """Add standard FIT analysis columns to activities table.
        
        Returns:
            Migration statistics
        """
        fit_columns = {
            # FIT file tracking
            "fit_file_path": "TEXT",
            "has_fit_analysis": "BOOLEAN",
            "fit_timeseries_file": "TEXT",
            "fit_data_points": "INTEGER",
            
            # Heart rate metrics from FIT
            "fit_hr_avg": "REAL",
            "fit_hr_max": "REAL",
            "fit_hr_min": "REAL",
            "fit_hr_std": "REAL",
            
            # Power metrics from FIT
            "fit_power_avg": "REAL",
            "fit_power_max": "REAL",
            "fit_power_normalized": "REAL",
            "fit_power_vi": "REAL",  # Variability Index
            
            # Cadence metrics from FIT
            "fit_cadence_avg": "REAL",
            "fit_cadence_max": "REAL",
            
            # Speed metrics from FIT
            "fit_speed_avg": "REAL",
            "fit_speed_max": "REAL",
            
            # Altitude metrics from FIT
            "fit_altitude_min": "REAL",
            "fit_altitude_max": "REAL",
            "fit_altitude_gain": "REAL",
            
            # Temperature from FIT
            "fit_temp_avg": "REAL",
            "fit_temp_min": "REAL",
            "fit_temp_max": "REAL",
            
            # Heart rate zones (percentage of time)
            "fit_hr_zone1_pct": "REAL",
            "fit_hr_zone2_pct": "REAL",
            "fit_hr_zone3_pct": "REAL",
            "fit_hr_zone4_pct": "REAL",
            "fit_hr_zone5_pct": "REAL",
            
            # Power zones (percentage of time)
            "fit_power_zone1_pct": "REAL",
            "fit_power_zone2_pct": "REAL",
            "fit_power_zone3_pct": "REAL",
            "fit_power_zone4_pct": "REAL",
            "fit_power_zone5_pct": "REAL",
            "fit_power_zone6_pct": "REAL",
            "fit_power_zone7_pct": "REAL",
            
            # Garmin-specific fields
            "garmin_activity_id": "TEXT",
            "start_date_gmt": "TIMESTAMP",
            "sport_type": "TEXT",
            "duration": "REAL",
            "elapsed_duration": "REAL",
            "moving_duration": "REAL",
            "elevation_loss": "REAL",
            "average_hr": "REAL",
            "max_hr": "REAL",
            "bmr_calories": "REAL",
            "average_power": "REAL",
            "max_power": "REAL",
            "normalized_power": "REAL",
            "training_stress_score": "REAL",
            "intensity_factor": "REAL",
            "avg_stride_length": "REAL",
            "vo2_max": "REAL",
            "lactate_threshold_hr": "REAL",
            "device_name": "TEXT",
            "max_temperature": "REAL",
            "min_temperature": "REAL",
            "start_latitude": "REAL",
            "start_longitude": "REAL",
            "end_latitude": "REAL",
            "end_longitude": "REAL",
            "source": "TEXT",
            "has_polyline": "BOOLEAN",
            "has_splits": "BOOLEAN",
            "pr_count": "INTEGER",
            "avg_ground_contact_time": "REAL",
            "avg_vertical_oscillation": "REAL",
            "avg_ground_contact_balance": "REAL",
            "avg_vertical_ratio": "REAL",
            "training_effect": "REAL",
            "anaerobic_training_effect": "REAL",
            "avg_left_balance": "REAL",
            "avg_right_balance": "REAL",
            "functional_threshold_power": "REAL",
            "threshold_power": "REAL",  # FTP from FIT files
            "avg_cadence": "REAL",  # Cadence metrics
            "max_cadence": "REAL",
            "avg_fractional_cadence": "REAL",
            "max_fractional_cadence": "REAL",
            "avg_temperature": "REAL",  # Temperature
            "left_right_balance": "REAL",  # Power balance
            "avg_left_torque_effectiveness": "REAL",
            "avg_right_torque_effectiveness": "REAL",
            "avg_left_pedal_smoothness": "REAL",
            "avg_right_pedal_smoothness": "REAL",
            "avg_left_power_phase": "TEXT",  # Store as JSON
            "avg_right_power_phase": "TEXT",
            "avg_left_power_phase_peak": "TEXT",
            "avg_right_power_phase_peak": "TEXT",
            "total_anaerobic_training_effect": "REAL",
            "total_training_effect": "REAL",
            "intensity_factor": "REAL",
            "avg_vam": "REAL",  # Vertical Ascent Meters/hour
            "total_work": "INTEGER",  # Joules
            "device_manufacturer": "TEXT",  # Device info
            "device_product": "TEXT",
            "device_serial_number": "TEXT",
            "device_software_version": "REAL",
            "lap_data": "TEXT",  # Store lap data as JSON
            "zones_config": "TEXT",  # Zone configuration
            "has_fit_data": "BOOLEAN",  # Track if FIT data was extracted
            "fit_parse_error": "TEXT",  # Store any FIT parsing errors
            "weather_temp": "REAL",
            "weather_humidity": "REAL",
            "weather_wind_speed": "REAL",
            "weather_wind_direction": "REAL",
            "splits_data": "TEXT",
            "hr_zones_data": "TEXT",
        }
        
        existing_columns = self.get_existing_columns("activities")
        added_columns = []
        
        with self.engine.connect() as conn:
            for col, sql_type in fit_columns.items():
                if col not in existing_columns:
                    try:
                        alter_sql = f"ALTER TABLE activities ADD COLUMN {col} {sql_type}"
                        conn.execute(text(alter_sql))
                        conn.commit()
                        added_columns.append(col)
                        logger.info(f"Added FIT column: {col} ({sql_type})")
                    except Exception as e:
                        logger.error(f"Failed to add column {col}: {e}")
        
        return {
            "added_columns": added_columns,
            "count": len(added_columns),
            "total_columns": len(existing_columns) + len(added_columns)
        }