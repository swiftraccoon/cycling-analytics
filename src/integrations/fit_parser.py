"""FIT file parser for extracting detailed second-by-second data."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import struct

import polars as pl
import numpy as np

logger = logging.getLogger(__name__)


class FITParser:
    """Parser for Garmin FIT files to extract detailed time-series data."""
    
    def __init__(self):
        """Initialize FIT parser."""
        self.has_fitparse = False
        try:
            import fitparse
            self.fitparse = fitparse
            self.has_fitparse = True
            logger.info("Using fitparse library for FIT file parsing")
        except ImportError:
            logger.warning("fitparse not installed. Install with: pip install fitparse")
    
    def parse_fit_file(self, fit_path: Path) -> Dict[str, Any]:
        """Parse a FIT file and extract all available data.
        
        Args:
            fit_path: Path to FIT file
            
        Returns:
            Dictionary containing parsed data including time-series records
        """
        if not self.has_fitparse:
            logger.error("fitparse library not available")
            return {}
        
        if not fit_path.exists():
            logger.error(f"FIT file not found: {fit_path}")
            return {}
        
        try:
            fitfile = self.fitparse.FitFile(str(fit_path))
            
            result = {
                "file_path": str(fit_path),
                "session_data": {},
                "lap_data": [],
                "record_data": [],  # Second-by-second data
                "hrv_data": [],
                "device_info": {},
                "zones": {},
                "events": [],
                "parsing_errors": []
            }
            
            # Parse all messages
            for message in fitfile.get_messages():
                try:
                    self._process_message(message, result)
                except Exception as e:
                    result["parsing_errors"].append(str(e))
            
            # Convert record data to DataFrame for easier analysis
            if result["record_data"]:
                result["records_df"] = self._create_records_dataframe(result["record_data"])
                result["summary_stats"] = self._calculate_summary_stats(result["records_df"])
            
            logger.info(f"Parsed FIT file with {len(result['record_data'])} data points")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing FIT file {fit_path}: {e}")
            return {"error": str(e)}
    
    def _process_message(self, message, result: Dict):
        """Process a single FIT message.
        
        Args:
            message: FIT message object
            result: Result dictionary to update
        """
        message_type = message.name
        
        if message_type == 'file_id':
            self._process_file_id(message, result)
        elif message_type == 'session':
            self._process_session(message, result)
        elif message_type == 'lap':
            self._process_lap(message, result)
        elif message_type == 'record':
            self._process_record(message, result)
        elif message_type == 'hrv':
            self._process_hrv(message, result)
        elif message_type == 'device_info':
            self._process_device_info(message, result)
        elif message_type == 'zones_target':
            self._process_zones(message, result)
        elif message_type == 'event':
            self._process_event(message, result)
    
    def _process_file_id(self, message, result: Dict):
        """Process file_id message."""
        device_info = {}
        for field in message:
            if field.name and field.value is not None:
                device_info[field.name] = field.value
        result["device_info"].update(device_info)
    
    def _process_session(self, message, result: Dict):
        """Process session message containing summary data."""
        session_data = {}
        for field in message:
            if field.name and field.value is not None:
                # Convert timestamps to datetime
                if field.name in ['timestamp', 'start_time']:
                    session_data[field.name] = self._convert_timestamp(field.value)
                else:
                    session_data[field.name] = field.value
        
        result["session_data"] = session_data
    
    def _process_lap(self, message, result: Dict):
        """Process lap message."""
        lap_data = {}
        for field in message:
            if field.name and field.value is not None:
                if field.name in ['timestamp', 'start_time']:
                    lap_data[field.name] = self._convert_timestamp(field.value)
                else:
                    lap_data[field.name] = field.value
        
        if lap_data:
            result["lap_data"].append(lap_data)
    
    def _process_record(self, message, result: Dict):
        """Process record message (second-by-second data)."""
        record = {}
        for field in message:
            if field.name and field.value is not None:
                if field.name == 'timestamp':
                    record[field.name] = self._convert_timestamp(field.value)
                elif field.name == 'left_right_balance':
                    # Handle fitparse bug: it incorrectly decodes raw value 128 as string "right"
                    # The raw value is what we actually want
                    if isinstance(field.value, str):
                        # fitparse bug - use raw_value if available
                        if hasattr(field, 'raw_value'):
                            # Raw value 128 typically means 0% or invalid reading
                            # FIT protocol uses 128-228 scale where 128=0%, 178=50%, 228=100%
                            if field.raw_value == 128:
                                record[field.name] = None  # Invalid/no balance data
                            else:
                                # Convert to percentage: (raw - 128) / 100 * 100
                                record[field.name] = field.raw_value - 128
                        else:
                            continue  # Skip if we can't get raw value
                    else:
                        # Numeric value is already decoded correctly by fitparse
                        # Values are in 128-228 range, convert to 0-100% scale
                        if field.value >= 128:
                            record[field.name] = field.value - 128
                        else:
                            record[field.name] = field.value  # Already in 0-100 scale
                else:
                    # Store raw value
                    record[field.name] = field.value
                    
                    # Handle special fields with units
                    if field.units:
                        record[f"{field.name}_units"] = field.units
        
        if record and 'timestamp' in record:
            result["record_data"].append(record)
    
    def _process_hrv(self, message, result: Dict):
        """Process HRV (Heart Rate Variability) message."""
        hrv_data = {}
        for field in message:
            if field.name and field.value is not None:
                hrv_data[field.name] = field.value
        
        if hrv_data:
            result["hrv_data"].append(hrv_data)
    
    def _process_device_info(self, message, result: Dict):
        """Process device info message."""
        for field in message:
            if field.name and field.value is not None:
                result["device_info"][field.name] = field.value
    
    def _process_zones(self, message, result: Dict):
        """Process zones message."""
        zones = {}
        for field in message:
            if field.name and field.value is not None:
                zones[field.name] = field.value
        
        if zones:
            result["zones"] = zones
    
    def _process_event(self, message, result: Dict):
        """Process event message."""
        event = {}
        for field in message:
            if field.name and field.value is not None:
                if field.name == 'timestamp':
                    event[field.name] = self._convert_timestamp(field.value)
                else:
                    event[field.name] = field.value
        
        if event:
            result["events"].append(event)
    
    def _convert_timestamp(self, timestamp):
        """Convert FIT timestamp to datetime.
        
        Args:
            timestamp: FIT timestamp
            
        Returns:
            datetime object
        """
        if isinstance(timestamp, datetime):
            return timestamp
        
        # FIT timestamps are seconds since Dec 31, 1989 00:00:00 UTC
        if isinstance(timestamp, (int, float)):
            from datetime import timedelta
            fit_epoch = datetime(1989, 12, 31, 0, 0, 0, tzinfo=timezone.utc)
            return fit_epoch + timedelta(seconds=int(timestamp))
        
        return timestamp
    
    def _create_records_dataframe(self, records: List[Dict]) -> pl.DataFrame:
        """Create a Polars DataFrame from record data.
        
        Args:
            records: List of record dictionaries
            
        Returns:
            Polars DataFrame with time-series data
        """
        if not records:
            return pl.DataFrame()
        
        # Convert to DataFrame
        df = pl.DataFrame(records)
        
        # Ensure timestamp column exists and is sorted
        if "timestamp" in df.columns:
            df = df.sort("timestamp")
            
            # Calculate elapsed time
            if not df.is_empty():
                start_time = df["timestamp"][0]
                df = df.with_columns(
                    ((pl.col("timestamp") - start_time).dt.total_seconds()).alias("elapsed_time")
                )
        
        # Convert units where applicable
        column_conversions = {
            "speed": ("m/s", "km/h", lambda x: x * 3.6),
            "distance": ("m", "km", lambda x: x / 1000),
            "altitude": ("m", "m", lambda x: x),
            "heart_rate": ("bpm", "bpm", lambda x: x),
            "cadence": ("rpm", "rpm", lambda x: x),
            "power": ("watts", "watts", lambda x: x),
            "temperature": ("C", "C", lambda x: x),
        }
        
        for col, (from_unit, to_unit, conversion) in column_conversions.items():
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).map_elements(conversion).alias(f"{col}_{to_unit}")
                )
        
        return df
    
    def _calculate_summary_stats(self, df: pl.DataFrame) -> Dict:
        """Calculate summary statistics from records DataFrame.
        
        Args:
            df: Records DataFrame
            
        Returns:
            Dictionary of summary statistics
        """
        stats = {}
        
        if df.is_empty():
            return stats
        
        # Time-based stats
        if "elapsed_time" in df.columns:
            stats["total_time"] = df["elapsed_time"].max()
            stats["recording_interval"] = df["elapsed_time"].diff().median()
        
        # Heart rate stats
        if "heart_rate" in df.columns:
            hr_data = df.filter(pl.col("heart_rate").is_not_null())["heart_rate"]
            if len(hr_data) > 0:
                stats["hr_avg"] = hr_data.mean()
                stats["hr_max"] = hr_data.max()
                stats["hr_min"] = hr_data.min()
                stats["hr_std"] = hr_data.std()
                
                # Calculate time in HR zones (assuming standard 5-zone model)
                if stats["hr_max"]:
                    zones = self._calculate_hr_zones(hr_data, stats["hr_max"])
                    stats["hr_zones"] = zones
        
        # Power stats
        if "power" in df.columns:
            power_data = df.filter(pl.col("power").is_not_null())["power"]
            if len(power_data) > 0:
                stats["power_avg"] = power_data.mean()
                stats["power_max"] = power_data.max()
                stats["power_normalized"] = self._calculate_normalized_power(power_data)
                stats["power_variability_index"] = stats["power_normalized"] / stats["power_avg"] if stats["power_avg"] > 0 else 0
                
                # Calculate power zones
                if "power_max" in stats:
                    zones = self._calculate_power_zones(power_data, stats.get("ftp", 250))
                    stats["power_zones"] = zones
        
        # Cadence stats
        if "cadence" in df.columns:
            cadence_data = df.filter(pl.col("cadence").is_not_null())["cadence"]
            if len(cadence_data) > 0:
                stats["cadence_avg"] = cadence_data.mean()
                stats["cadence_max"] = cadence_data.max()
        
        # Speed/pace stats
        if "speed_km/h" in df.columns:
            speed_data = df.filter(pl.col("speed_km/h").is_not_null())["speed_km/h"]
            if len(speed_data) > 0:
                stats["speed_avg"] = speed_data.mean()
                stats["speed_max"] = speed_data.max()
        
        # Altitude stats
        if "altitude" in df.columns:
            alt_data = df.filter(pl.col("altitude").is_not_null())["altitude"]
            if len(alt_data) > 0:
                stats["altitude_min"] = alt_data.min()
                stats["altitude_max"] = alt_data.max()
                stats["altitude_gain"] = self._calculate_elevation_gain(alt_data)
        
        # Temperature stats
        if "temperature" in df.columns:
            temp_data = df.filter(pl.col("temperature").is_not_null())["temperature"]
            if len(temp_data) > 0:
                stats["temp_avg"] = temp_data.mean()
                stats["temp_min"] = temp_data.min()
                stats["temp_max"] = temp_data.max()
        
        return stats
    
    def _calculate_normalized_power(self, power_data: pl.Series) -> float:
        """Calculate Normalized Power (NP).
        
        Args:
            power_data: Series of power values
            
        Returns:
            Normalized power value
        """
        if len(power_data) < 30:
            return power_data.mean() if len(power_data) > 0 else 0
        
        # Convert to numpy for rolling calculations
        power_array = power_data.to_numpy()
        
        # 30-second rolling average
        window_size = 30
        rolling_avg = np.convolve(power_array, np.ones(window_size)/window_size, mode='valid')
        
        # Raise to 4th power, average, then take 4th root
        np_value = np.power(np.mean(np.power(rolling_avg, 4)), 0.25)
        
        return float(np_value)
    
    def _calculate_hr_zones(self, hr_data: pl.Series, max_hr: float) -> Dict:
        """Calculate time in heart rate zones.
        
        Args:
            hr_data: Series of heart rate values
            max_hr: Maximum heart rate
            
        Returns:
            Dictionary with time in each zone
        """
        zones = {
            "zone1": 0,  # < 60% max HR
            "zone2": 0,  # 60-70% max HR
            "zone3": 0,  # 70-80% max HR
            "zone4": 0,  # 80-90% max HR
            "zone5": 0,  # > 90% max HR
        }
        
        for hr in hr_data:
            pct = (hr / max_hr) * 100
            if pct < 60:
                zones["zone1"] += 1
            elif pct < 70:
                zones["zone2"] += 1
            elif pct < 80:
                zones["zone3"] += 1
            elif pct < 90:
                zones["zone4"] += 1
            else:
                zones["zone5"] += 1
        
        # Convert to percentages
        total = sum(zones.values())
        if total > 0:
            for zone in zones:
                zones[zone] = (zones[zone] / total) * 100
        
        return zones
    
    def _calculate_power_zones(self, power_data: pl.Series, ftp: float) -> Dict:
        """Calculate time in power zones based on FTP.
        
        Args:
            power_data: Series of power values
            ftp: Functional Threshold Power
            
        Returns:
            Dictionary with time in each zone
        """
        zones = {
            "zone1": 0,  # Active Recovery < 55% FTP
            "zone2": 0,  # Endurance 56-75% FTP
            "zone3": 0,  # Tempo 76-90% FTP
            "zone4": 0,  # Threshold 91-105% FTP
            "zone5": 0,  # VO2Max 106-120% FTP
            "zone6": 0,  # Anaerobic 121-150% FTP
            "zone7": 0,  # Neuromuscular > 150% FTP
        }
        
        for power in power_data:
            pct = (power / ftp) * 100
            if pct < 55:
                zones["zone1"] += 1
            elif pct <= 75:
                zones["zone2"] += 1
            elif pct <= 90:
                zones["zone3"] += 1
            elif pct <= 105:
                zones["zone4"] += 1
            elif pct <= 120:
                zones["zone5"] += 1
            elif pct <= 150:
                zones["zone6"] += 1
            else:
                zones["zone7"] += 1
        
        # Convert to percentages
        total = sum(zones.values())
        if total > 0:
            for zone in zones:
                zones[zone] = (zones[zone] / total) * 100
        
        return zones
    
    def _calculate_elevation_gain(self, altitude_data: pl.Series) -> float:
        """Calculate total elevation gain.
        
        Args:
            altitude_data: Series of altitude values
            
        Returns:
            Total elevation gain in meters
        """
        if len(altitude_data) < 2:
            return 0
        
        alt_array = altitude_data.to_numpy()
        diff = np.diff(alt_array)
        
        # Only count positive differences (gains)
        # Apply a small threshold to filter out noise
        threshold = 1.0  # meters
        gains = diff[diff > threshold]
        
        return float(np.sum(gains)) if len(gains) > 0 else 0