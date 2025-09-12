#!/usr/bin/env python3
"""
Example script demonstrating the complete cycling analytics data pipeline.

This script shows how to use all the core pipeline components together
to process CSV files from intervals.icu exports.
"""

import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.pipeline_orchestrator import DataPipelineOrchestrator, PipelineConfig
from src.data.ingestion_tracker import FileMovementStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline_execution.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Execute the complete data pipeline."""
    
    # Define paths relative to project root
    project_root = Path(__file__).parent.parent
    
    config = PipelineConfig(
        database_path=project_root / "data" / "cycling_analytics.db",
        incoming_directory=project_root / "data" / "bronze" / "incoming",
        archive_directory=project_root / "data" / "bronze" / "archive",
        file_movement_strategy=FileMovementStrategy.MOVE,
        skip_validation=False,
        skip_deduplication=False,
        batch_size=1000
    )
    
    logger.info("Starting cycling analytics data pipeline")
    logger.info(f"Database: {config.database_path}")
    logger.info(f"Incoming: {config.incoming_directory}")
    logger.info(f"Archive: {config.archive_directory}")
    
    # Create and execute pipeline
    orchestrator = DataPipelineOrchestrator(config)
    results = orchestrator.execute_pipeline()
    
    # Generate and display report
    report = orchestrator.create_pipeline_report(results)
    print("\n" + "="*80)
    print(report)
    print("="*80 + "\n")
    
    # Log final summary
    logger.info(
        f"Pipeline execution completed: {results.success_rate:.1f}% success rate, "
        f"{results.total_records_imported:,} records imported"
    )
    
    # Exit with appropriate code
    if results.failed_files > 0:
        logger.warning(f"Pipeline completed with {results.failed_files} file failures")
        sys.exit(1)
    else:
        logger.info("Pipeline execution successful")
        sys.exit(0)


if __name__ == "__main__":
    main()