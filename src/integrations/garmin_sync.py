"""Script to sync activities from Garmin Connect with FIT file downloads.

Usage:
    python scripts/sync_garmin.py --email YOUR_EMAIL --password YOUR_PASSWORD
    
    Or set environment variables:
    export GARMIN_EMAIL=your_email
    export GARMIN_PASSWORD=your_password
    python scripts/sync_garmin.py
    
    For detailed FIT file analysis:
    python scripts/sync_garmin.py --analyze-fit
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import polars as pl

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from src.integrations.garmin_connect import GarminConnectClient, GarminConnectSyncManager
from src.integrations.fit_parser import FITParser
from src.data.deduplicator import Deduplicator
from src.storage.database.manager import DatabaseManager
from src.config import (
    DATABASE_PATH, BRONZE_DIR, GARMIN_STATE_FILE, GARMIN_FIT_DIR,
    GARMIN_EMAIL, GARMIN_PASSWORD, GARMIN_TOKENSTORE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sync cycling activities from Garmin Connect"
    )
    
    parser.add_argument(
        "--email",
        help="Garmin Connect email. Can also use GARMIN_EMAIL env var",
        default=GARMIN_EMAIL
    )
    
    parser.add_argument(
        "--password",
        help="Garmin Connect password. Can also use GARMIN_PASSWORD env var",
        default=GARMIN_PASSWORD
    )
    
    parser.add_argument(
        "--start-date",
        help="Start date for sync (YYYY-MM-DD). Default: 30 days ago",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d")
    )
    
    parser.add_argument(
        "--end-date",
        help="End date for sync (YYYY-MM-DD). Default: today",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d")
    )
    
    parser.add_argument(
        "--force-full",
        help="Force full sync instead of incremental",
        action="store_true"
    )
    
    parser.add_argument(
        "--no-fit",
        help="Skip downloading FIT files",
        action="store_true"
    )
    
    parser.add_argument(
        "--analyze-fit",
        help="Perform detailed FIT file analysis",
        action="store_true"
    )
    
    parser.add_argument(
        "--dry-run",
        help="Fetch data but don't save to database",
        action="store_true"
    )
    
    parser.add_argument(
        "--export-csv",
        help="Export fetched activities to CSV file",
        type=Path
    )
    
    parser.add_argument(
        "--limit",
        help="Maximum number of activities to sync",
        type=int,
        default=100
    )
    
    return parser.parse_args()


def analyze_fit_files(fit_dir: Path, activity_df: pl.DataFrame) -> pl.DataFrame:
    """Analyze FIT files and add detailed metrics to activities.
    
    Args:
        fit_dir: Directory containing FIT files
        activity_df: DataFrame with activities
        
    Returns:
        Enhanced DataFrame with FIT analysis data
    """
    parser = FITParser()
    
    if not parser.has_fitparse:
        logger.warning("fitparse library not installed. Skipping FIT analysis.")
        logger.info("Install with: uv pip install fitparse")
        return activity_df
    
    enhanced_activities = []
    successful_parses = 0
    failed_parses = 0
    
    for row in activity_df.iter_rows(named=True):
        activity = dict(row)
        
        # Check if we have a FIT file for this activity
        if "fit_file_path" in activity and activity["fit_file_path"]:
            fit_path = Path(activity["fit_file_path"])
            
            if fit_path.exists():
                logger.info(f"Analyzing FIT file: {fit_path.name}")
                
                try:
                    # Parse FIT file with simplified parser
                    fit_data = parser.parse_fit_file(fit_path)
                    
                    if "error" not in fit_data:
                        # Extract session data and add to activity
                        if "session_data" in fit_data and fit_data["session_data"]:
                            session = fit_data["session_data"]
                            
                            # Map critical FIT fields directly (no fit_ prefix needed)
                            if session.get("threshold_power"):
                                activity["threshold_power"] = session["threshold_power"]
                            if session.get("avg_cadence"):
                                activity["avg_cadence"] = session["avg_cadence"]
                            if session.get("max_cadence"):
                                activity["max_cadence"] = session["max_cadence"]
                            if session.get("avg_temperature"):
                                activity["avg_temperature"] = session["avg_temperature"]
                            if session.get("max_temperature"):
                                activity["max_temperature"] = session["max_temperature"]
                            if session.get("left_right_balance"):
                                activity["left_right_balance"] = session["left_right_balance"]
                            if session.get("intensity_factor"):
                                activity["intensity_factor"] = session["intensity_factor"]
                            if session.get("training_stress_score"):
                                activity["training_stress_score"] = session["training_stress_score"]
                            if session.get("total_work"):
                                activity["total_work"] = session["total_work"]
                            
                            # Power dynamics
                            for field in ["avg_left_torque_effectiveness", "avg_right_torque_effectiveness",
                                        "avg_left_pedal_smoothness", "avg_right_pedal_smoothness"]:
                                if session.get(field):
                                    activity[field] = session[field]
                            
                            # Training effects
                            if session.get("total_training_effect"):
                                activity["total_training_effect"] = session["total_training_effect"]
                            if session.get("total_anaerobic_training_effect"):
                                activity["total_anaerobic_training_effect"] = session["total_anaerobic_training_effect"]
                        
                        # Store device info
                        if "device_info" in fit_data and fit_data["device_info"]:
                            device = fit_data["device_info"]
                            if device.get("manufacturer"):
                                activity["device_manufacturer"] = device["manufacturer"]
                            if device.get("product"):
                                activity["device_product"] = device["product"]
                            if device.get("serial_number"):
                                activity["device_serial_number"] = str(device["serial_number"])
                        
                        # Store lap data as JSON
                        if "lap_data" in fit_data and fit_data["lap_data"]:
                            import json
                            activity["lap_data"] = json.dumps(fit_data["lap_data"])
                        
                        # Mark as having FIT analysis (but column doesn't exist yet)
                        fit_fields_added = len([k for k in activity.keys() if k in [
                            "threshold_power", "avg_cadence", "avg_temperature", "left_right_balance"
                        ]])
                        activity["has_fit_data"] = fit_fields_added > 0
                        
                        logger.info(f"  - Extracted {activity.get('fit_data_points', 0)} data points")
                        logger.info(f"  - Added FIT data: FTP={activity.get('threshold_power')}, Cadence={activity.get('avg_cadence')}")
                        
                        # Save time-series data if available
                        if "records_df" in fit_data and fit_data["records_df"] is not None:
                            if not fit_data["records_df"].is_empty():
                                # Save time-series data to parquet for efficient storage
                                activity_id = activity.get('garmin_activity_id', activity.get('id'))
                                ts_file = fit_dir / "timeseries" / f"{activity_id}.parquet"
                                ts_file.parent.mkdir(exist_ok=True)
                                fit_data["records_df"].write_parquet(ts_file)
                                activity["fit_timeseries_file"] = str(ts_file)
                                logger.info(f"  - Saved time-series data to {ts_file.name}")
                        
                        successful_parses += 1
                    else:
                        logger.warning(f"  - Error parsing FIT file: {fit_data.get('error')}")
                        activity["has_fit_data"] = False
                        activity["fit_parse_error"] = fit_data.get('error')
                        failed_parses += 1
                        
                except Exception as e:
                    logger.error(f"  - Exception parsing FIT file: {str(e)}")
                    activity["has_fit_data"] = False
                    activity["fit_parse_error"] = str(e)
                    failed_parses += 1
            else:
                logger.warning(f"FIT file not found: {fit_path}")
                activity["has_fit_data"] = False
                activity["fit_parse_error"] = "File not found"
        else:
            activity["has_fit_data"] = False
        
        enhanced_activities.append(activity)
    
    logger.info(f"FIT parsing complete: {successful_parses} successful, {failed_parses} failed")
    return pl.DataFrame(enhanced_activities)


def main():
    """Main sync function."""
    args = parse_arguments()
    
    # Validate required parameters
    if not args.email or not args.password:
        logger.error("Both --email and --password are required")
        logger.error("You can also set GARMIN_EMAIL and GARMIN_PASSWORD environment variables")
        sys.exit(1)
    
    # Initialize client and sync manager
    logger.info(f"Initializing Garmin Connect client for: {args.email}")
    
    try:
        client = GarminConnectClient(
            email=args.email,
            password=args.password,
            tokenstore=GARMIN_TOKENSTORE
        )
    except Exception as e:
        logger.error(f"Failed to connect to Garmin Connect: {e}")
        sys.exit(1)
    
    sync_manager = GarminConnectSyncManager(
        client=client,
        state_file=GARMIN_STATE_FILE,
        fit_storage_dir=GARMIN_FIT_DIR
    )
    
    # Get current sync status
    status = sync_manager.get_sync_status()
    if status["last_sync_date"]:
        logger.info(f"Last sync: {status['last_sync_date']} ({status['last_sync_count']} activities)")
        logger.info(f"Total synced: {status['total_synced_activities']} activities")
        logger.info(f"FIT files: {status['fit_files_count']} in {status['fit_files_directory']}")
    else:
        logger.info("No previous sync found, will perform full sync")
    
    # Perform sync
    logger.info("Starting sync from Garmin Connect...")
    activities_df = sync_manager.sync_activities(
        start_date=args.start_date,
        end_date=args.end_date,
        download_fit=not args.no_fit,
        force_full=args.force_full,
        limit=args.limit
    )
    
    if activities_df.is_empty():
        logger.info("No new activities to process")
        return
    
    logger.info(f"Fetched {len(activities_df)} activities from Garmin Connect")
    
    # Analyze FIT files if requested
    if args.analyze_fit and not args.no_fit:
        logger.info("Performing detailed FIT file analysis...")
        activities_df = analyze_fit_files(GARMIN_FIT_DIR, activities_df)
        
        # Show summary of FIT analysis
        if "has_fit_data" in activities_df.columns:
            fit_analyzed = activities_df.filter(pl.col("has_fit_data") == True).shape[0]
            logger.info(f"Successfully analyzed {fit_analyzed} FIT files")
        else:
            logger.info("FIT analysis results not tracked")
    
    # Export to CSV if requested
    if args.export_csv:
        logger.info(f"Exporting activities to {args.export_csv}")
        activities_df.write_csv(args.export_csv)
        logger.info(f"Exported {len(activities_df)} activities to CSV")
    
    if args.dry_run:
        logger.info("Dry run mode - not saving to database")
        
        # Show sample of data
        logger.info("\nSample of fetched activities:")
        sample_cols = ["garmin_activity_id", "start_date_local", "name", "type", "distance", "duration", "average_power"]
        available_cols = [col for col in sample_cols if col in activities_df.columns]
        print(activities_df.select(available_cols).head(5))
        
        if args.analyze_fit:
            # Show FIT analysis sample
            fit_cols = ["fit_power_normalized", "fit_power_vi", "fit_hr_avg", "fit_data_points"]
            available_fit_cols = [col for col in fit_cols if col in activities_df.columns]
            if available_fit_cols:
                logger.info("\nFIT analysis sample:")
                print(activities_df.select(["garmin_activity_id"] + available_fit_cols).head(5))
        
        return
    
    # Save to bronze layer as CSV for archival
    bronze_file = BRONZE_DIR / "incoming" / f"garmin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    bronze_file.parent.mkdir(parents=True, exist_ok=True)
    activities_df.write_csv(bronze_file)
    logger.info(f"Saved raw data to bronze layer: {bronze_file}")
    
    # Deduplicate
    logger.info("Deduplicating activities...")
    deduplicator = Deduplicator()
    
    # If we have existing data in database, include it for deduplication
    db_path = Path(DATABASE_PATH)
    if db_path.exists():
        db = DatabaseManager()
        existing_activities = db.get_activities()
        
        if not existing_activities.is_empty():
            logger.info(f"Including {len(existing_activities)} existing activities for deduplication")
            
            # Align schemas before concatenation
            # Get union of all columns
            all_columns = list(set(activities_df.columns) | set(existing_activities.columns))
            all_columns.sort()  # Consistent ordering
            
            # Create aligned DataFrames with matching schemas
            new_cols = []
            for col in all_columns:
                if col in activities_df.columns:
                    new_cols.append(activities_df[col])
                else:
                    # Create null column with appropriate type
                    if col in existing_activities.columns:
                        dtype = existing_activities[col].dtype
                    else:
                        dtype = pl.Utf8  # Default to string
                    new_cols.append(pl.lit(None, dtype=dtype).alias(col))
            activities_df_aligned = pl.DataFrame(new_cols)
            
            existing_cols = []
            for col in all_columns:
                if col in existing_activities.columns:
                    existing_cols.append(existing_activities[col])
                else:
                    # Create null column with matching type
                    if col in activities_df.columns:
                        dtype = activities_df[col].dtype
                    else:
                        dtype = pl.Utf8  # Default to string
                    existing_cols.append(pl.Series(col, [None] * len(existing_activities), dtype=dtype))
            existing_activities_aligned = pl.DataFrame(existing_cols)
            
            # Now combine with matching schemas
            combined_df = pl.concat([activities_df_aligned, existing_activities_aligned])
            deduped_df, dedup_report = deduplicator.deduplicate(combined_df)
            
            # Get only the new unique activities
            existing_ids = set(existing_activities["id"].to_list())
            new_activities = deduped_df.filter(~pl.col("id").is_in(existing_ids))
        else:
            new_activities, dedup_report = deduplicator.deduplicate(activities_df)
    else:
        new_activities, dedup_report = deduplicator.deduplicate(activities_df)
    
    logger.info(f"Deduplication complete: {len(new_activities)} unique new activities")
    logger.info(f"  - Exact duplicates removed: {dedup_report.get('exact_duplicates', 0)}")
    logger.info(f"  - Near duplicates removed: {dedup_report.get('near_duplicates', 0)}")
    
    if new_activities.is_empty():
        logger.info("No new unique activities to save")
        return
    
    # Save to database
    logger.info("Saving to database...")
    db = DatabaseManager()
    save_stats = db.save_activities(new_activities)
    
    logger.info(f"Database save complete:")
    logger.info(f"  - Saved: {save_stats['saved']} activities")
    logger.info(f"  - Skipped (duplicates): {save_stats.get('skipped', 0)}")
    if 'errors' in save_stats:
        logger.info(f"  - Errors: {save_stats['errors']}")
    
    # Get updated totals
    all_activities = db.get_activities()
    logger.info(f"Total activities in database: {len(all_activities)}")
    
    # Show summary of new activities
    if save_stats["saved"] > 0:
        logger.info("\nSummary of new activities:")
        recent = new_activities.head(5)
        sample_cols = ["start_date_local", "name", "type", "distance", "average_power", "normalized_power"]
        available_cols = [col for col in sample_cols if col in recent.columns]
        
        for row in recent.select(available_cols).iter_rows(named=True):
            date = row.get("start_date_local", "N/A")
            if isinstance(date, datetime):
                date = date.strftime("%Y-%m-%d")
            name = row.get("name", "N/A")[:50]
            activity_type = row.get("type", "N/A")
            distance = row.get("distance", 0)
            distance_km = distance / 1000 if distance else 0
            avg_power = row.get("average_power", "N/A")
            np_power = row.get("normalized_power", "N/A")
            
            print(f"  {date} - {name} ({activity_type}) - {distance_km:.1f}km - AP: {avg_power}W, NP: {np_power}W")
    
    # Summary of FIT analysis if performed
    if args.analyze_fit and not args.no_fit:
        # Check for activities with FIT data
        if "threshold_power" in all_activities.columns:
            fit_activities = all_activities.filter(pl.col("threshold_power").is_not_null())
            if not fit_activities.is_empty():
                logger.info(f"\nTotal activities with FIT data (FTP): {len(fit_activities)}")
                
                # Show FTP summary
                ftp_values = fit_activities["threshold_power"].unique().sort()
                logger.info(f"Unique FTP values found: {ftp_values.to_list()}")
        
        if "avg_cadence" in all_activities.columns:
            cadence_activities = all_activities.filter(pl.col("avg_cadence").is_not_null())
            if not cadence_activities.is_empty():
                logger.info(f"Activities with cadence data: {len(cadence_activities)}")
        
        # Show power metrics summary if available
        if "normalized_power" in all_activities.columns:
            power_activities = all_activities.filter(pl.col("normalized_power").is_not_null())
            if not power_activities.is_empty():
                avg_np = power_activities["normalized_power"].mean()
                logger.info(f"Average Normalized Power: {avg_np:.0f}W")


if __name__ == "__main__":
    main()