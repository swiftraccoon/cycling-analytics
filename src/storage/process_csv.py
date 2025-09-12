"""Process CSV files from the bronze layer."""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.etl.extractors.csv_extractor import CSVExtractor
from src.data.deduplicator import Deduplicator
from src.storage.database.manager import DatabaseManager
from src.config import BRONZE_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Process CSV files from bronze/incoming directory."""
    logger.info("Starting CSV processing...")
    
    # Extract data from CSV files
    extractor = CSVExtractor(bronze_path=str(BRONZE_DIR))
    df, metadata = extractor.extract_all(include_archive=False)
    
    if df.is_empty():
        logger.info("No CSV files found in bronze/incoming/")
        return
    
    logger.info(f"Extracted {len(df)} activities from {len(metadata)} files")
    
    # Deduplicate
    deduplicator = Deduplicator()
    df_dedup, dedup_report = deduplicator.deduplicate(df)
    
    logger.info(f"Deduplication complete:")
    logger.info(f"  - Total records: {dedup_report['total_records']}")
    logger.info(f"  - Unique records: {dedup_report['unique_records']}")
    logger.info(f"  - Exact duplicates: {dedup_report.get('exact_duplicates', 0)}")
    logger.info(f"  - Near duplicates: {dedup_report.get('near_duplicates', 0)}")
    
    # Save to database (uses default path from config)
    db = DatabaseManager()
    save_stats = db.save_activities(df_dedup)
    
    logger.info(f"Database save complete:")
    logger.info(f"  - Saved: {save_stats['saved']} activities")
    logger.info(f"  - Skipped (duplicates): {save_stats.get('skipped', 0)}")
    if 'errors' in save_stats:
        logger.info(f"  - Errors: {save_stats['errors']}")
    
    # Archive processed files
    for file_info in metadata:
        source_file = Path(file_info["file_path"])
        if source_file.exists():
            extractor.move_to_archive(source_file)
            logger.info(f"Archived: {source_file.name}")
    
    # Get total count
    all_activities = db.get_activities()
    logger.info(f"Total activities in database: {len(all_activities)}")


if __name__ == "__main__":
    main()