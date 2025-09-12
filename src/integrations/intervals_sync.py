"""Script to sync activities from Intervals.icu.

Usage:
    python scripts/sync_intervals_icu.py --athlete-id YOUR_ID --api-key YOUR_KEY
    
    Or set environment variables:
    export INTERVALS_ICU_ATHLETE_ID=your_id
    export INTERVALS_ICU_API_KEY=your_key
    python scripts/sync_intervals_icu.py
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import polars as pl

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from src.integrations.intervals_icu import IntervalsICUClient, IntervalsICUSyncManager
from src.etl.extractors.csv_extractor import CSVExtractor
from src.data.deduplicator import Deduplicator
from src.storage.database.manager import DatabaseManager
from src.config import DATABASE_PATH, BRONZE_DIR, INTERVALS_ICU_STATE_FILE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sync cycling activities from Intervals.icu"
    )
    
    parser.add_argument(
        "--athlete-id",
        help="Athlete ID (username or numeric ID). Can also use INTERVALS_ICU_ATHLETE_ID env var",
        default=os.environ.get("INTERVALS_ICU_ATHLETE_ID")
    )
    
    parser.add_argument(
        "--api-key",
        help="API key from Settings -> Developer Settings. Can also use INTERVALS_ICU_API_KEY env var",
        default=os.environ.get("INTERVALS_ICU_API_KEY")
    )
    
    parser.add_argument(
        "--data-dir",
        help="Data directory path",
        default="data",
        type=Path
    )
    
    parser.add_argument(
        "--force-full",
        help="Force full sync instead of incremental",
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
    
    return parser.parse_args()


def main():
    """Main sync function."""
    args = parse_arguments()
    
    # Validate required parameters
    if not args.athlete_id or not args.api_key:
        logger.error("Both --athlete-id and --api-key are required")
        logger.error("You can also set INTERVALS_ICU_ATHLETE_ID and INTERVALS_ICU_API_KEY environment variables")
        sys.exit(1)
    
    # Setup paths (use config defaults unless overridden)
    if args.data_dir != Path("data"):
        # User specified a custom data dir
        bronze_dir = Path(args.data_dir) / "bronze"
        bronze_dir.mkdir(parents=True, exist_ok=True)
        state_file = Path(args.data_dir) / "intervals_icu_sync_state.json"
        db_path = Path(args.data_dir) / "silver" / "cycling_analytics.db"
    else:
        # Use defaults from config
        bronze_dir = BRONZE_DIR
        state_file = INTERVALS_ICU_STATE_FILE
        db_path = Path(DATABASE_PATH)
    
    # Initialize client and sync manager
    logger.info(f"Initializing Intervals.icu client for athlete: {args.athlete_id}")
    client = IntervalsICUClient(args.athlete_id, args.api_key)
    sync_manager = IntervalsICUSyncManager(client, state_file)
    
    # Get current sync status
    status = sync_manager.get_sync_status()
    if status["last_sync_date"]:
        logger.info(f"Last sync: {status['last_sync_date']} ({status['last_sync_count']} activities)")
    else:
        logger.info("No previous sync found, will perform full sync")
    
    # Perform sync
    logger.info("Starting sync from Intervals.icu...")
    activities_df = sync_manager.sync_activities(force_full=args.force_full)
    
    if activities_df.is_empty():
        logger.info("No new activities to process")
        return
    
    logger.info(f"Fetched {len(activities_df)} activities from Intervals.icu")
    
    # Export to CSV if requested
    if args.export_csv:
        logger.info(f"Exporting activities to {args.export_csv}")
        activities_df.write_csv(args.export_csv)
        logger.info(f"Exported {len(activities_df)} activities to CSV")
    
    if args.dry_run:
        logger.info("Dry run mode - not saving to database")
        # Show sample of data
        logger.info("\nSample of fetched activities:")
        sample_cols = ["id", "start_date_local", "name", "type", "distance", "moving_time", "icu_training_load"]
        available_cols = [col for col in sample_cols if col in activities_df.columns]
        print(activities_df.select(available_cols).head(5))
        return
    
    # Save to bronze layer as CSV for archival
    bronze_file = bronze_dir / "incoming" / f"intervals_icu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    bronze_file.parent.mkdir(parents=True, exist_ok=True)
    activities_df.write_csv(bronze_file)
    logger.info(f"Saved raw data to bronze layer: {bronze_file}")
    
    # Deduplicate
    logger.info("Deduplicating activities...")
    deduplicator = Deduplicator()
    
    # If we have existing data in database, include it for deduplication
    if db_path.exists():
        db = DatabaseManager()  # Uses default from config
        existing_activities = db.get_activities()
        
        if not existing_activities.is_empty():
            logger.info(f"Including {len(existing_activities)} existing activities for deduplication")
            combined_df = pl.concat([activities_df, existing_activities])
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
    db = DatabaseManager()  # Uses default from config
    save_stats = db.save_activities(new_activities)
    
    logger.info(f"Database save complete:")
    logger.info(f"  - Saved: {save_stats['saved']} activities")
    logger.info(f"  - Skipped (duplicates): {save_stats['skipped']}")
    logger.info(f"  - Errors: {save_stats['errors']}")
    
    # Get updated totals
    all_activities = db.get_activities()
    logger.info(f"Total activities in database: {len(all_activities)}")
    
    # Show summary of new activities
    if save_stats["saved"] > 0:
        logger.info("\nSummary of new activities:")
        recent = new_activities.head(5)
        sample_cols = ["start_date_local", "name", "type", "distance", "icu_training_load"]
        available_cols = [col for col in sample_cols if col in recent.columns]
        
        for row in recent.select(available_cols).iter_rows(named=True):
            date = row.get("start_date_local", "N/A")
            if isinstance(date, datetime):
                date = date.strftime("%Y-%m-%d")
            name = row.get("name", "N/A")[:50]
            activity_type = row.get("type", "N/A")
            distance = row.get("distance", 0)
            distance_km = distance / 1000 if distance else 0
            load = row.get("icu_training_load", "N/A")
            
            print(f"  {date} - {name} ({activity_type}) - {distance_km:.1f}km - Load: {load}")


if __name__ == "__main__":
    main()