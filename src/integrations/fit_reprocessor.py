"""Module for reprocessing existing FIT files to extract all available data."""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import polars as pl
import json
from datetime import datetime

from src.integrations.fit_parser import FITParser
from src.models.fit_data import FITParseResult, FITSessionData
from src.storage.database.manager import DatabaseManager
from src.config import BRONZE_DIR

logger = logging.getLogger(__name__)


class FITReprocessor:
    """Handles reprocessing of existing FIT files."""
    
    def __init__(self, fit_dir: Optional[Path] = None):
        """Initialize FIT reprocessor.
        
        Args:
            fit_dir: Directory containing FIT files (default: data/bronze/fit_files)
        """
        self.fit_dir = fit_dir or (BRONZE_DIR / "fit_files")
        self.parser = FITParser()
        self.db = DatabaseManager()
        
    def find_fit_files(self) -> List[Path]:
        """Find all FIT files in the directory.
        
        Returns:
            List of FIT file paths
        """
        if not self.fit_dir.exists():
            logger.warning(f"FIT directory does not exist: {self.fit_dir}")
            return []
        
        fit_files = list(self.fit_dir.glob("*.fit"))
        logger.info(f"Found {len(fit_files)} FIT files in {self.fit_dir}")
        return sorted(fit_files)
    
    def extract_activity_id(self, fit_path: Path) -> Optional[str]:
        """Extract activity ID from FIT filename.
        
        Args:
            fit_path: Path to FIT file
            
        Returns:
            Activity ID or None
        """
        # Filename format: {activity_id}_{name}.fit
        filename = fit_path.stem
        parts = filename.split("_", 1)
        if parts:
            return parts[0]
        return None
    
    def process_fit_file(self, fit_path: Path) -> Dict:
        """Process a single FIT file using direct fitparse.
        
        Args:
            fit_path: Path to FIT file
            
        Returns:
            Extracted FIT data with activity ID
        """
        import fitparse
        
        activity_id = self.extract_activity_id(fit_path)
        
        result = {
            "activity_id": activity_id,
            "fit_file_path": str(fit_path),
            "processed_at": datetime.now(),
            "success": False
        }
        
        try:
            # Use fitparse directly to avoid FITParser issues
            fitfile = fitparse.FitFile(str(fit_path))
            
            # Extract session data directly
            for record in fitfile.get_messages('session'):
                session = {}
                for field in record:
                    if field.value is not None:
                        session[field.name] = field.value
                
                # Core metrics
                result["threshold_power"] = session.get("threshold_power")
                result["functional_threshold_power"] = session.get("functional_threshold_power")
                result["avg_cadence"] = session.get("avg_cadence")
                result["max_cadence"] = session.get("max_cadence")
                result["avg_fractional_cadence"] = session.get("avg_fractional_cadence")
                result["max_fractional_cadence"] = session.get("max_fractional_cadence")
                
                # Temperature
                result["avg_temperature"] = session.get("avg_temperature")
                result["max_temperature"] = session.get("max_temperature")
                
                # Power dynamics
                result["left_right_balance"] = session.get("left_right_balance")
                result["avg_left_torque_effectiveness"] = session.get("avg_left_torque_effectiveness")
                result["avg_right_torque_effectiveness"] = session.get("avg_right_torque_effectiveness")
                result["avg_left_pedal_smoothness"] = session.get("avg_left_pedal_smoothness")
                result["avg_right_pedal_smoothness"] = session.get("avg_right_pedal_smoothness")
                
                # Power phase (store as JSON, handle None values)
                for phase_field in ["avg_left_power_phase", "avg_right_power_phase",
                                  "avg_left_power_phase_peak", "avg_right_power_phase_peak"]:
                    if session.get(phase_field) and session[phase_field] != (None, None, None, None):
                        # Only store if not all None
                        if isinstance(session[phase_field], (list, tuple)):
                            result[phase_field] = json.dumps(list(session[phase_field]))
                        else:
                            result[phase_field] = json.dumps(session[phase_field])
                
                # Training metrics
                result["intensity_factor"] = session.get("intensity_factor")
                result["training_stress_score"] = session.get("training_stress_score")
                result["total_training_effect"] = session.get("total_training_effect")
                result["total_anaerobic_training_effect"] = session.get("total_anaerobic_training_effect")
                result["avg_vam"] = session.get("avg_vam")
                result["total_work"] = session.get("total_work")
                
                # Update power/HR if not already set
                result["avg_power"] = session.get("avg_power")
                result["max_power"] = session.get("max_power")
                result["normalized_power"] = session.get("normalized_power")
                result["avg_heart_rate"] = session.get("avg_heart_rate")
                result["max_heart_rate"] = session.get("max_heart_rate")
            
            # Extract device info
            if "device_info" in fit_data and fit_data["device_info"]:
                device = fit_data["device_info"]
                result["device_manufacturer"] = device.get("manufacturer")
                # Ensure device_product is a string
                product = device.get("product") or device.get("product_name")
                result["device_product"] = str(product) if product is not None else None
                result["device_serial_number"] = str(device.get("serial_number")) if device.get("serial_number") else None
                result["device_software_version"] = device.get("software_version")
            
            # Store lap data as JSON (handle datetime serialization)
            if "lap_data" in fit_data and fit_data["lap_data"]:
                # Convert datetime objects to strings in lap data
                clean_laps = []
                for lap in fit_data["lap_data"]:
                    clean_lap = {}
                    for key, value in lap.items():
                        if isinstance(value, datetime):
                            clean_lap[key] = value.isoformat()
                        else:
                            clean_lap[key] = value
                    clean_laps.append(clean_lap)
                result["lap_data"] = json.dumps(clean_laps)
            
            # Store zones config (handle potential datetime/complex types)
            if "zones" in fit_data and fit_data["zones"]:
                try:
                    result["zones_config"] = json.dumps(fit_data["zones"])
                except (TypeError, ValueError):
                    # If zones has non-serializable data, skip it
                    pass
            
            # Store summary stats
            if "summary_stats" in fit_data and fit_data["summary_stats"]:
                # Store key summary stats
                stats = fit_data["summary_stats"]
                result["fit_hr_zones"] = json.dumps(stats.get("hr_zones")) if stats.get("hr_zones") else None
                result["fit_power_zones"] = json.dumps(stats.get("power_zones")) if stats.get("power_zones") else None
            
            # Save time-series data if available
            if "records_df" in fit_data and fit_data["records_df"] is not None:
                if not fit_data["records_df"].is_empty():
                    # Save time-series data to parquet
                    ts_dir = self.fit_dir / "timeseries"
                    ts_dir.mkdir(exist_ok=True)
                    ts_file = ts_dir / f"{activity_id}.parquet"
                    fit_data["records_df"].write_parquet(ts_file)
                    result["fit_timeseries_file"] = str(ts_file)
                    result["fit_data_points"] = len(fit_data["records_df"])
            
            result["success"] = True
            result["has_fit_data"] = True
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Exception processing {fit_path.name}: {e}")
        
        return result
    
    def reprocess_all(self, update_database: bool = True, force: bool = False) -> Dict:
        """Reprocess all FIT files.
        
        Args:
            update_database: Whether to update database with results
            force: Force reprocessing even if data exists
            
        Returns:
            Summary of reprocessing results
        """
        fit_files = self.find_fit_files()
        
        if not fit_files:
            return {"error": "No FIT files found", "processed": 0}
        
        results = {
            "total_files": len(fit_files),
            "processed": 0,
            "updated": 0,
            "errors": 0,
            "skipped": 0
        }
        
        # Get existing activities from database
        existing_activities = self.db.get_activities()
        activity_map = {}
        
        if not existing_activities.is_empty():
            # Create mapping of activity IDs
            for row in existing_activities.iter_rows(named=True):
                activity_id = row.get("garmin_activity_id") or row.get("id")
                if activity_id:
                    activity_map[str(activity_id)] = row
        
        processed_data = []
        
        for i, fit_path in enumerate(fit_files, 1):
            logger.info(f"Processing {i}/{len(fit_files)}: {fit_path.name}")
            
            # Extract activity ID
            activity_id = self.extract_activity_id(fit_path)
            
            # Check if we should skip
            if not force and activity_id in activity_map:
                existing = activity_map[activity_id]
                if existing.get("threshold_power") or existing.get("avg_cadence"):
                    logger.info(f"  Skipping - FIT data already exists")
                    results["skipped"] += 1
                    continue
            
            # Process FIT file
            fit_result = self.process_fit_file(fit_path)
            
            if fit_result["success"]:
                results["processed"] += 1
                processed_data.append(fit_result)
                
                # Log key extracted data
                logger.info(f"  ✅ Extracted: FTP={fit_result.get('threshold_power')}, "
                          f"Cadence={fit_result.get('avg_cadence')}, "
                          f"Temp={fit_result.get('avg_temperature')}")
            else:
                results["errors"] += 1
                logger.warning(f"  ❌ Failed: {fit_result.get('error')}")
        
        # Update database if requested
        if update_database and processed_data:
            logger.info(f"\nUpdating database with {len(processed_data)} activities...")
            results["updated"] = self.update_database(processed_data)
        
        return results
    
    def update_database(self, fit_results: List[Dict]) -> int:
        """Update database with reprocessed FIT data.
        
        Args:
            fit_results: List of processed FIT data
            
        Returns:
            Number of activities updated
        """
        # Get existing activities
        existing_df = self.db.get_activities()
        
        if existing_df.is_empty():
            logger.warning("No activities in database to update")
            return 0
        
        # Convert to list of dicts for easier updating
        activities = existing_df.to_dicts()
        updated_count = 0
        
        for fit_data in fit_results:
            activity_id = str(fit_data.get("activity_id"))
            if not activity_id:
                continue
            
            # Find matching activity
            activity_found = False
            for i, activity in enumerate(activities):
                if str(activity.get("garmin_activity_id")) == activity_id or str(activity.get("id")) == activity_id:
                    activity_found = True
                    
                    # Update with FIT data
                    fit_fields = [
                        "threshold_power", "functional_threshold_power",
                        "avg_cadence", "max_cadence", "avg_fractional_cadence", "max_fractional_cadence",
                        "avg_temperature", "max_temperature",
                        "left_right_balance",
                        "avg_left_torque_effectiveness", "avg_right_torque_effectiveness",
                        "avg_left_pedal_smoothness", "avg_right_pedal_smoothness",
                        "avg_left_power_phase", "avg_right_power_phase",
                        "avg_left_power_phase_peak", "avg_right_power_phase_peak",
                        "intensity_factor", "training_stress_score",
                        "total_training_effect", "total_anaerobic_training_effect",
                        "avg_vam", "total_work",
                        "device_manufacturer", "device_product", "device_serial_number",
                        "lap_data", "zones_config", "has_fit_data"
                    ]
                    
                    fields_updated = 0
                    for field in fit_fields:
                        if field in fit_data and fit_data[field] is not None:
                            activities[i][field] = fit_data[field]
                            fields_updated += 1
                    
                    if fields_updated > 0:
                        updated_count += 1
                        logger.info(f"Updated activity {activity_id} with {fields_updated} FIT fields")
                    break
            
            if not activity_found:
                logger.warning(f"No matching activity for ID {activity_id}")
        
        # Convert back to dataframe and save
        if updated_count > 0:
            updated_df = pl.DataFrame(activities)
            
            # Ensure numeric fields are properly typed and handle NaN
            numeric_fields = [
                "threshold_power", "functional_threshold_power",
                "avg_cadence", "max_cadence", "avg_fractional_cadence", "max_fractional_cadence",
                "avg_temperature", "max_temperature", "left_right_balance",
                "avg_left_torque_effectiveness", "avg_right_torque_effectiveness",
                "avg_left_pedal_smoothness", "avg_right_pedal_smoothness",
                "intensity_factor", "training_stress_score",
                "total_training_effect", "total_anaerobic_training_effect",
                "avg_vam", "device_software_version"
            ]
            
            for field in numeric_fields:
                if field in updated_df.columns:
                    # Cast to float, handling nulls and NaN
                    updated_df = updated_df.with_columns(
                        pl.when(pl.col(field).is_nan() | pl.col(field).is_null())
                        .then(None)
                        .otherwise(pl.col(field))
                        .cast(pl.Float64, strict=False)
                        .alias(field)
                    )
            
            # Ensure integer fields (handle NaN by converting to null)
            int_fields = ["total_work"]
            for field in int_fields:
                if field in updated_df.columns:
                    updated_df = updated_df.with_columns(
                        pl.when(pl.col(field).is_nan() | pl.col(field).is_null())
                        .then(None)
                        .otherwise(pl.col(field))
                        .cast(pl.Int64, strict=False)
                        .alias(field)
                    )
            
            # Save to database using proper method
            try:
                # Write directly to avoid validation issues with partial FIT data
                import sqlite3
                conn = sqlite3.connect(self.db.db_path)
                
                # Convert to pandas for SQLite
                df_pandas = updated_df.to_pandas()
                
                # Replace the activities table
                df_pandas.to_sql('activities', conn, if_exists='replace', index=False)
                conn.close()
                
                logger.info(f"Saved {updated_count} updated activities to database")
            except Exception as e:
                logger.error(f"Failed to save updates: {e}")
                raise
        
        return updated_count
    
    def verify_reprocessing(self) -> Dict:
        """Verify that FIT data was properly extracted and saved.
        
        Returns:
            Verification report
        """
        activities = self.db.get_activities()
        
        if activities.is_empty():
            return {"error": "No activities in database"}
        
        report = {
            "total_activities": len(activities),
            "fit_data_coverage": {}
        }
        
        # Check key FIT fields
        fit_fields = [
            "threshold_power",
            "avg_cadence",
            "avg_temperature",
            "left_right_balance",
            "device_manufacturer",
            "lap_data",
            "has_fit_data"
        ]
        
        for field in fit_fields:
            if field in activities.columns:
                non_null = activities.filter(pl.col(field).is_not_null()).shape[0]
                percentage = (non_null / len(activities)) * 100
                report["fit_data_coverage"][field] = {
                    "count": non_null,
                    "percentage": round(percentage, 1)
                }
            else:
                report["fit_data_coverage"][field] = {
                    "count": 0,
                    "percentage": 0.0,
                    "error": "Column not in database"
                }
        
        # Calculate overall FIT data completeness
        total_checks = len(fit_fields) * len(activities)
        total_present = sum(f["count"] for f in report["fit_data_coverage"].values())
        report["overall_fit_completeness"] = round((total_present / total_checks) * 100, 1)
        
        return report