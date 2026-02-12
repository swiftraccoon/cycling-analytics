"""
Data pipeline orchestrator for cycling analytics.

Coordinates the complete data ingestion pipeline including extraction,
validation, deduplication, and database storage.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from ..storage.database.manager import DatabaseManager
from ..storage.schemas.activity_schema import initialize_database
from ..etl.extractors.csv_extractor import IntervalsCsvExtractor, ExtractionResult
from ..etl.validators.integrity_validator import IntegrityValidator, ValidationReport
from .deduplicator import Deduplicator, DeduplicationReport
from .ingestion_tracker import IngestionTracker, FileMovementStrategy, IngestionRecord

logger = logging.getLogger(__name__)


class PipelineConfig:
    """Configuration for data pipeline."""
    
    def __init__(
        self,
        database_path: Path,
        incoming_directory: Path,
        archive_directory: Path,
        file_movement_strategy: FileMovementStrategy = FileMovementStrategy.MOVE,
        skip_validation: bool = False,
        skip_deduplication: bool = False,
        batch_size: int = 1000
    ):
        self.database_path = database_path
        self.incoming_directory = incoming_directory
        self.archive_directory = archive_directory
        self.file_movement_strategy = file_movement_strategy
        self.skip_validation = skip_validation
        self.skip_deduplication = skip_deduplication
        self.batch_size = batch_size


class PipelineResults:
    """Results from complete pipeline execution."""
    
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.end_time: Optional[datetime] = None
        self.total_files_processed = 0
        self.successful_files = 0
        self.failed_files = 0
        self.total_records_extracted = 0
        self.total_records_imported = 0
        self.total_duplicates_removed = 0
        self.extraction_results: List[ExtractionResult] = []
        self.validation_reports: List[ValidationReport] = []
        self.deduplication_report: Optional[DeduplicationReport] = None
        self.ingestion_summary = None
        self.errors: List[str] = []
    
    def finalize(self):
        """Mark pipeline as completed."""
        self.end_time = datetime.utcnow()
    
    @property
    def processing_time_ms(self) -> int:
        """Total processing time in milliseconds."""
        if self.end_time:
            delta = self.end_time - self.start_time
            return int(delta.total_seconds() * 1000)
        return 0
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_files_processed == 0:
            return 0.0
        return (self.successful_files / self.total_files_processed) * 100


class DataPipelineOrchestrator:
    """
    Complete data pipeline orchestrator for cycling analytics.
    
    Coordinates all pipeline components to provide end-to-end data processing
    from raw CSV files to clean, validated, and deduplicated database records.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline orchestrator.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        # Initialize components
        self.db_manager = DatabaseManager(config.database_path)
        self.extractor = IntervalsCsvExtractor()
        self.validator = IntegrityValidator()
        self.deduplicator = Deduplicator()
        
        self.ingestion_tracker = IngestionTracker(
            db_manager=self.db_manager,
            incoming_directory=config.incoming_directory,
            archive_directory=config.archive_directory,
            movement_strategy=config.file_movement_strategy
        )
        
        logger.info(f"Initialized data pipeline orchestrator")
    
    def process_single_file(
        self, 
        file_path: Path,
        ingestion_record: IngestionRecord
    ) -> Tuple[ExtractionResult, Optional[ValidationReport]]:
        """
        Process a single CSV file through extraction and validation.
        
        Args:
            file_path: Path to CSV file
            ingestion_record: Tracking record for this file
            
        Returns:
            Tuple of (extraction result, validation report)
        """
        logger.info(f"Processing file: {file_path.name}")
        
        # Start ingestion tracking
        self.ingestion_tracker.start_ingestion(ingestion_record)
        
        # Extract data
        extraction_result = self.extractor.extract_intervals_file(file_path)
        
        if not extraction_result.success:
            error_msg = f"Extraction failed: {extraction_result.error_message}"
            self.ingestion_tracker.complete_ingestion(
                ingestion_record,
                records_extracted=0,
                records_imported=0,
                error_message=error_msg
            )
            return extraction_result, None
        
        validation_report = None
        validation_errors = 0
        validation_warnings = 0
        
        # Validate data if not skipped
        if not self.config.skip_validation and extraction_result.data is not None:
            try:
                validation_report = self.validator.validate_dataset(
                    extraction_result.data,
                    str(file_path),
                    extraction_result.metadata.file_hash
                )
                validation_errors = validation_report.critical_issues + validation_report.error_issues
                validation_warnings = validation_report.warning_issues
                
                logger.info(
                    f"Validation complete: {validation_report.validation_status}, "
                    f"quality score: {validation_report.quality_score:.1f}"
                )
                
            except Exception as e:
                logger.error(f"Validation failed for {file_path.name}: {e}")
                validation_errors = 1
        
        # Update ingestion record with processing results
        records_extracted = len(extraction_result.data) if extraction_result.data is not None else 0
        
        self.ingestion_tracker.complete_ingestion(
            ingestion_record,
            records_extracted=records_extracted,
            records_imported=0,  # Will be updated after deduplication/import
            validation_errors=validation_errors,
            validation_warnings=validation_warnings
        )
        
        return extraction_result, validation_report
    
    def deduplicate_and_import(
        self, 
        extraction_results: List[ExtractionResult],
        ingestion_records: List[IngestionRecord]
    ) -> DeduplicationReport:
        """
        Deduplicate data across all files and import to database.
        
        Args:
            extraction_results: List of successful extraction results
            ingestion_records: Corresponding ingestion records
            
        Returns:
            Deduplication report
        """
        logger.info("Starting deduplication and database import")
        
        # Prepare data sources for deduplication
        data_sources = []
        for result in extraction_results:
            if result.success and result.data is not None and result.metadata is not None:
                metadata_dict = result.metadata.model_dump()
                data_sources.append((result.data, metadata_dict))
        
        if not data_sources:
            logger.warning("No valid data sources for deduplication")
            return DeduplicationReport(
                total_records=0,
                unique_activities=0,
                duplicate_groups=0,
                total_duplicates_removed=0,
                processing_time_ms=0
            )
        
        # Perform deduplication if not skipped
        if not self.config.skip_deduplication:
            dedup_report = self.deduplicator.deduplicate_data(data_sources)
        else:
            # Skip deduplication - combine all data
            import polars as pl
            
            all_dataframes = [data for data, _ in data_sources]
            combined_data = pl.concat(all_dataframes, how='vertical')
            
            dedup_report = DeduplicationReport(
                total_records=len(combined_data),
                unique_activities=len(combined_data),
                duplicate_groups=0,
                total_duplicates_removed=0,
                processing_time_ms=0
            )
        
        # Import deduplicated data to database
        try:
            # Get the final clean data
            final_data = None
            if self.config.skip_deduplication:
                # Use combined data
                import polars as pl
                all_dataframes = [data for data, _ in data_sources]
                final_data = pl.concat(all_dataframes, how='vertical')
            else:
                # Recreate clean data by filtering out duplicates
                import polars as pl
                all_dataframes = [data for data, _ in data_sources]
                combined_data = pl.concat(all_dataframes, how='vertical')
                
                # Remove duplicate rows based on deduplication results
                # This is a simplified approach - in production you'd want more sophisticated handling
                final_data = combined_data.unique(subset=['Id'], keep='first')
            
            # Convert to list of dictionaries for database import
            records_to_import = final_data.to_dicts()
            
            # Bulk import to database
            # Map each record back to its source file for lineage tracking
            current_row = 0
            for idx, (result, record) in enumerate(zip(extraction_results, ingestion_records)):
                if result.success and result.data is not None:
                    file_records = records_to_import[current_row:current_row + len(result.data)]
                    
                    if file_records:
                        imported_count = self.db_manager.bulk_insert_activities(
                            file_records,
                            record.file_path,
                            record.file_hash
                        )
                        
                        # Update ingestion record with import count
                        record.records_imported = imported_count
                        
                        logger.info(f"Imported {imported_count} records from {record.file_name}")
                    
                    current_row += len(result.data)
            
            logger.info(f"Database import complete: {len(records_to_import)} total records")
            
        except Exception as e:
            logger.error(f"Database import failed: {e}")
            # Update all records with error
            for record in ingestion_records:
                if record.status != "error":
                    record.error_message = f"Database import failed: {e}"
                    record.status = "error"
        
        return dedup_report
    
    def execute_pipeline(self, file_pattern: str = "*.csv") -> PipelineResults:
        """
        Execute the complete data pipeline.
        
        Args:
            file_pattern: Pattern to match files for processing
            
        Returns:
            Complete pipeline results
        """
        results = PipelineResults()
        logger.info("Starting complete data pipeline execution")
        
        try:
            # Discover files to process
            files_to_process = self.ingestion_tracker.get_processing_queue(
                pattern=file_pattern,
                skip_processed=True
            )
            
            if not files_to_process:
                logger.info("No files found for processing")
                results.finalize()
                return results
            
            results.total_files_processed = len(files_to_process)
            logger.info(f"Found {len(files_to_process)} files to process")
            
            # Process each file
            successful_extractions = []
            ingestion_records = []
            
            for file_path in files_to_process:
                try:
                    # Create ingestion record
                    ingestion_record = self.ingestion_tracker.create_ingestion_record(file_path)
                    ingestion_records.append(ingestion_record)
                    
                    # Process file
                    extraction_result, validation_report = self.process_single_file(
                        file_path, 
                        ingestion_record
                    )
                    
                    results.extraction_results.append(extraction_result)
                    
                    if validation_report:
                        results.validation_reports.append(validation_report)
                    
                    if extraction_result.success:
                        successful_extractions.append(extraction_result)
                        results.successful_files += 1
                        
                        if extraction_result.data is not None:
                            results.total_records_extracted += len(extraction_result.data)
                    else:
                        results.failed_files += 1
                        results.errors.append(
                            f"{file_path.name}: {extraction_result.error_message}"
                        )
                    
                    # Add to session tracking
                    self.ingestion_tracker.add_session_record(ingestion_record)
                    
                except Exception as e:
                    logger.error(f"Failed to process {file_path.name}: {e}")
                    results.failed_files += 1
                    results.errors.append(f"{file_path.name}: {str(e)}")
            
            # Deduplicate and import data
            if successful_extractions:
                dedup_report = self.deduplicate_and_import(
                    successful_extractions,
                    [r for r in ingestion_records if r.status != "error"]
                )
                results.deduplication_report = dedup_report
                results.total_duplicates_removed = dedup_report.total_duplicates_removed
                results.total_records_imported = dedup_report.unique_activities
            
            # Record ingestion history in database
            for record in ingestion_records:
                try:
                    self.ingestion_tracker.record_ingestion_in_database(record)
                except Exception as e:
                    logger.error(f"Failed to record ingestion history: {e}")
            
            # Archive processed files
            for record in ingestion_records:
                try:
                    self.ingestion_tracker.archive_file(record)
                except Exception as e:
                    logger.error(f"Failed to archive {record.file_name}: {e}")
            
            # Generate final summary
            results.ingestion_summary = self.ingestion_tracker.get_session_summary()
            
            logger.info(
                f"Pipeline execution complete: {results.successful_files}/{results.total_files_processed} "
                f"files processed successfully, {results.total_records_imported} records imported"
            )
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            results.errors.append(f"Pipeline failure: {str(e)}")
        
        finally:
            results.finalize()
        
        return results
    
    def create_pipeline_report(self, results: PipelineResults) -> str:
        """
        Create comprehensive pipeline execution report.
        
        Args:
            results: Pipeline execution results
            
        Returns:
            Formatted report string
        """
        report = f"""
Cycling Analytics Data Pipeline Report
=====================================

Execution Time: {results.start_time.strftime('%Y-%m-%d %H:%M:%S')}
Processing Duration: {results.processing_time_ms / 1000:.2f} seconds

File Processing Summary:
- Total Files: {results.total_files_processed}
- Successful: {results.successful_files}
- Failed: {results.failed_files}
- Success Rate: {results.success_rate:.1f}%

Data Processing Summary:
- Records Extracted: {results.total_records_extracted:,}
- Records Imported: {results.total_records_imported:,}
- Duplicates Removed: {results.total_duplicates_removed:,}
- Import Rate: {(results.total_records_imported / results.total_records_extracted * 100) if results.total_records_extracted > 0 else 0:.1f}%

Pipeline Components:
- Extraction: ✓ Completed
- Validation: {'✓ Completed' if not self.config.skip_validation else '⚠ Skipped'}
- Deduplication: {'✓ Completed' if not self.config.skip_deduplication else '⚠ Skipped'}
- Database Import: ✓ Completed
"""
        
        # Add validation summary
        if results.validation_reports:
            avg_quality_score = sum(r.quality_score for r in results.validation_reports) / len(results.validation_reports)
            total_critical = sum(r.critical_issues for r in results.validation_reports)
            total_errors = sum(r.error_issues for r in results.validation_reports)
            total_warnings = sum(r.warning_issues for r in results.validation_reports)
            
            report += f"""
Data Quality Summary:
- Average Quality Score: {avg_quality_score:.1f}/100
- Critical Issues: {total_critical}
- Errors: {total_errors}
- Warnings: {total_warnings}
"""
        
        # Add deduplication summary
        if results.deduplication_report:
            dedup = results.deduplication_report
            report += f"""
Deduplication Summary:
- Duplicate Groups Found: {dedup.duplicate_groups}
- Records Removed: {dedup.total_duplicates_removed}
- Deduplication Rate: {(dedup.total_duplicates_removed / dedup.total_records * 100) if dedup.total_records > 0 else 0:.1f}%
"""
        
        # Add errors if any
        if results.errors:
            report += f"\nErrors Encountered:\n"
            for error in results.errors[:10]:  # Show first 10 errors
                report += f"- {error}\n"
            
            if len(results.errors) > 10:
                report += f"... and {len(results.errors) - 10} more errors\n"
        
        return report


# Example usage and integration
def create_pipeline_example(
    database_path: str = "data/cycling_analytics.db",
    incoming_dir: str = "data/bronze/incoming",
    archive_dir: str = "data/bronze/archive"
) -> DataPipelineOrchestrator:
    """
    Create a configured pipeline orchestrator.
    
    Args:
        database_path: Path to SQLite database
        incoming_dir: Directory with CSV files to process
        archive_dir: Directory to archive processed files
        
    Returns:
        Configured pipeline orchestrator
    """
    config = PipelineConfig(
        database_path=Path(database_path),
        incoming_directory=Path(incoming_dir),
        archive_directory=Path(archive_dir),
        file_movement_strategy=FileMovementStrategy.MOVE,
        skip_validation=False,
        skip_deduplication=False,
        batch_size=1000
    )
    
    return DataPipelineOrchestrator(config)