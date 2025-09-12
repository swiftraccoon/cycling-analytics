"""
File ingestion tracking system for cycling analytics.

Tracks all imported files with comprehensive metadata, manages file movement
from incoming to archive directories, and generates detailed ingestion reports.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import hashlib
from pydantic import BaseModel, Field
from enum import Enum

from ..storage.database.manager import DatabaseManager
from ..etl.extractors.csv_extractor import FileMetadata

logger = logging.getLogger(__name__)


class IngestionStatus(str, Enum):
    """Status of file ingestion process."""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    ARCHIVED = "archived"


class FileMovementStrategy(str, Enum):
    """Strategy for moving files after ingestion."""
    MOVE = "move"          # Move file to archive
    COPY = "copy"          # Copy file to archive, keep original
    LEAVE = "leave"        # Leave file in place
    DELETE = "delete"      # Delete after successful ingestion


class IngestionRecord(BaseModel):
    """Record of a single file ingestion."""
    
    file_path: str
    file_name: str
    file_size: int
    file_hash: str
    
    # Ingestion metadata
    ingestion_id: Optional[int] = None
    status: IngestionStatus = IngestionStatus.PENDING
    start_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    processing_time_ms: int = 0
    
    # Processing results
    records_extracted: int = 0
    records_imported: int = 0
    duplicates_found: int = 0
    validation_errors: int = 0
    validation_warnings: int = 0
    
    # Error information
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # Archive information
    archived: bool = False
    archive_path: Optional[str] = None
    archive_timestamp: Optional[datetime] = None


class IngestionSummary(BaseModel):
    """Summary of ingestion session."""
    
    session_id: str
    session_timestamp: datetime
    total_files_processed: int
    successful_ingestions: int
    failed_ingestions: int
    total_records_imported: int
    total_duplicates_found: int
    total_processing_time_ms: int
    files_archived: int
    
    # Detailed records
    ingestion_records: List[IngestionRecord] = Field(default_factory=list)


class IngestionTracker:
    """
    Comprehensive file ingestion tracking system.
    
    Manages the complete lifecycle of file ingestion including tracking,
    validation, archiving, and reporting.
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        incoming_directory: Path,
        archive_directory: Path,
        movement_strategy: FileMovementStrategy = FileMovementStrategy.MOVE
    ):
        """
        Initialize ingestion tracker.
        
        Args:
            db_manager: Database manager instance
            incoming_directory: Directory containing files to process
            archive_directory: Directory for processed files
            movement_strategy: How to handle files after processing
        """
        self.db_manager = db_manager
        self.incoming_directory = Path(incoming_directory)
        self.archive_directory = Path(archive_directory)
        self.movement_strategy = movement_strategy
        
        # Ensure directories exist
        self.incoming_directory.mkdir(parents=True, exist_ok=True)
        self.archive_directory.mkdir(parents=True, exist_ok=True)
        
        # Current session tracking
        self.current_session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.session_records: List[IngestionRecord] = []
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate SHA-256 hash of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hexadecimal hash string
        """
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            raise
    
    def create_ingestion_record(self, file_path: Path) -> IngestionRecord:
        """
        Create initial ingestion record for a file.
        
        Args:
            file_path: Path to file being ingested
            
        Returns:
            Initial ingestion record
        """
        try:
            stat = file_path.stat()
            file_hash = self.calculate_file_hash(file_path)
            
            record = IngestionRecord(
                file_path=str(file_path.absolute()),
                file_name=file_path.name,
                file_size=stat.st_size,
                file_hash=file_hash,
                status=IngestionStatus.PENDING,
                start_timestamp=datetime.utcnow()
            )
            
            logger.debug(f"Created ingestion record for {file_path.name}")
            return record
            
        except Exception as e:
            logger.error(f"Failed to create ingestion record for {file_path}: {e}")
            raise
    
    def is_file_already_processed(self, file_hash: str) -> bool:
        """
        Check if file was already processed based on hash.
        
        Args:
            file_hash: SHA-256 hash of file
            
        Returns:
            True if file was already processed
        """
        try:
            return self.db_manager.is_file_already_ingested(file_hash)
        except Exception as e:
            logger.error(f"Failed to check if file was processed: {e}")
            return False
    
    def start_ingestion(self, record: IngestionRecord) -> None:
        """
        Mark ingestion as started.
        
        Args:
            record: Ingestion record to update
        """
        record.status = IngestionStatus.PROCESSING
        record.start_timestamp = datetime.utcnow()
        logger.info(f"Started ingestion of {record.file_name}")
    
    def complete_ingestion(
        self,
        record: IngestionRecord,
        records_extracted: int,
        records_imported: int,
        duplicates_found: int = 0,
        validation_errors: int = 0,
        validation_warnings: int = 0,
        error_message: Optional[str] = None
    ) -> None:
        """
        Mark ingestion as completed.
        
        Args:
            record: Ingestion record to update
            records_extracted: Number of records extracted from file
            records_imported: Number of records successfully imported
            duplicates_found: Number of duplicate records found
            validation_errors: Number of validation errors
            validation_warnings: Number of validation warnings
            error_message: Error message if ingestion failed
        """
        record.end_timestamp = datetime.utcnow()
        
        if record.start_timestamp:
            time_diff = record.end_timestamp - record.start_timestamp
            record.processing_time_ms = int(time_diff.total_seconds() * 1000)
        
        record.records_extracted = records_extracted
        record.records_imported = records_imported
        record.duplicates_found = duplicates_found
        record.validation_errors = validation_errors
        record.validation_warnings = validation_warnings
        
        # Determine final status
        if error_message:
            record.status = IngestionStatus.ERROR
            record.error_message = error_message
        elif validation_errors > 0:
            record.status = IngestionStatus.ERROR
        elif validation_warnings > 0:
            record.status = IngestionStatus.WARNING
        else:
            record.status = IngestionStatus.SUCCESS
        
        logger.info(
            f"Completed ingestion of {record.file_name}: "
            f"{records_imported}/{records_extracted} records imported, "
            f"status: {record.status}"
        )
    
    def record_ingestion_in_database(self, record: IngestionRecord) -> None:
        """
        Store ingestion record in database.
        
        Args:
            record: Completed ingestion record
        """
        try:
            validation_errors = None
            if record.error_message:
                validation_errors = {
                    'error_message': record.error_message,
                    'validation_errors': record.validation_errors,
                    'validation_warnings': record.validation_warnings
                }
            
            record.ingestion_id = self.db_manager.record_file_ingestion(
                file_path=record.file_path,
                file_name=record.file_name,
                file_size=record.file_size,
                file_hash=record.file_hash,
                records_count=record.records_extracted,
                duplicates_found=record.duplicates_found,
                validation_status=record.status.value,
                validation_errors=validation_errors,
                processing_time_ms=record.processing_time_ms
            )
            
            logger.debug(f"Recorded ingestion in database: ID {record.ingestion_id}")
            
        except Exception as e:
            logger.error(f"Failed to record ingestion in database: {e}")
            # Don't raise - ingestion was successful, database recording failed
    
    def archive_file(self, record: IngestionRecord) -> bool:
        """
        Archive processed file based on configured strategy.
        
        Args:
            record: Ingestion record for file to archive
            
        Returns:
            True if file was successfully archived
        """
        if record.status not in [IngestionStatus.SUCCESS, IngestionStatus.WARNING]:
            logger.info(f"Skipping archive of failed ingestion: {record.file_name}")
            return False
        
        source_path = Path(record.file_path)
        
        if not source_path.exists():
            logger.warning(f"Source file not found for archiving: {source_path}")
            return False
        
        # Create archive path with timestamp to avoid conflicts
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        archive_name = f"{timestamp}_{source_path.name}"
        archive_path = self.archive_directory / archive_name
        
        try:
            if self.movement_strategy == FileMovementStrategy.MOVE:
                shutil.move(str(source_path), str(archive_path))
                logger.info(f"Moved {record.file_name} to archive")
                
            elif self.movement_strategy == FileMovementStrategy.COPY:
                shutil.copy2(str(source_path), str(archive_path))
                logger.info(f"Copied {record.file_name} to archive")
                
            elif self.movement_strategy == FileMovementStrategy.DELETE:
                source_path.unlink()
                logger.info(f"Deleted {record.file_name} after processing")
                return True  # No archive path to set
                
            else:  # LEAVE
                logger.info(f"Left {record.file_name} in original location")
                return True
            
            # Update record with archive information
            record.archived = True
            record.archive_path = str(archive_path)
            record.archive_timestamp = datetime.utcnow()
            
            # Update database
            try:
                self.db_manager.update_archive_status(record.file_hash, archived=True)
            except Exception as e:
                logger.error(f"Failed to update archive status in database: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to archive {record.file_name}: {e}")
            return False
    
    def discover_files(self, pattern: str = "*.csv") -> List[Path]:
        """
        Discover files in incoming directory.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List of file paths to process
        """
        try:
            files = list(self.incoming_directory.glob(pattern))
            logger.info(f"Discovered {len(files)} files matching '{pattern}'")
            return sorted(files)
        except Exception as e:
            logger.error(f"Failed to discover files: {e}")
            return []
    
    def get_processing_queue(
        self, 
        pattern: str = "*.csv",
        skip_processed: bool = True
    ) -> List[Path]:
        """
        Get list of files ready for processing.
        
        Args:
            pattern: File pattern to match
            skip_processed: Whether to skip already processed files
            
        Returns:
            List of files ready for processing
        """
        all_files = self.discover_files(pattern)
        
        if not skip_processed:
            return all_files
        
        # Filter out already processed files
        ready_files = []
        for file_path in all_files:
            try:
                file_hash = self.calculate_file_hash(file_path)
                if not self.is_file_already_processed(file_hash):
                    ready_files.append(file_path)
                else:
                    logger.info(f"Skipping already processed file: {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to check processing status for {file_path}: {e}")
                # Include file in queue despite error
                ready_files.append(file_path)
        
        logger.info(f"Found {len(ready_files)} files ready for processing")
        return ready_files
    
    def add_session_record(self, record: IngestionRecord) -> None:
        """
        Add record to current session.
        
        Args:
            record: Ingestion record to add
        """
        self.session_records.append(record)
    
    def get_session_summary(self) -> IngestionSummary:
        """
        Get summary of current ingestion session.
        
        Returns:
            Complete session summary
        """
        successful = sum(1 for r in self.session_records if r.status == IngestionStatus.SUCCESS)
        failed = len(self.session_records) - successful
        
        total_records = sum(r.records_imported for r in self.session_records)
        total_duplicates = sum(r.duplicates_found for r in self.session_records)
        total_time = sum(r.processing_time_ms for r in self.session_records)
        files_archived = sum(1 for r in self.session_records if r.archived)
        
        return IngestionSummary(
            session_id=self.current_session_id,
            session_timestamp=datetime.utcnow(),
            total_files_processed=len(self.session_records),
            successful_ingestions=successful,
            failed_ingestions=failed,
            total_records_imported=total_records,
            total_duplicates_found=total_duplicates,
            total_processing_time_ms=total_time,
            files_archived=files_archived,
            ingestion_records=self.session_records.copy()
        )
    
    def get_ingestion_history(
        self, 
        limit: Optional[int] = None,
        status_filter: Optional[IngestionStatus] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical ingestion records from database.
        
        Args:
            limit: Maximum number of records to return
            status_filter: Filter by ingestion status
            
        Returns:
            List of historical ingestion records
        """
        try:
            history = self.db_manager.get_ingestion_history(limit)
            
            if status_filter:
                history = [
                    record for record in history 
                    if record.get('validation_status') == status_filter.value
                ]
            
            return history
        except Exception as e:
            logger.error(f"Failed to get ingestion history: {e}")
            return []
    
    def create_ingestion_report(self) -> str:
        """
        Create human-readable ingestion report for current session.
        
        Returns:
            Formatted ingestion report
        """
        summary = self.get_session_summary()
        
        report = f"""
File Ingestion Report
====================

Session: {summary.session_id}
Timestamp: {summary.session_timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Summary:
- Files Processed: {summary.total_files_processed}
- Successful: {summary.successful_ingestions}
- Failed: {summary.failed_ingestions}
- Success Rate: {(summary.successful_ingestions / summary.total_files_processed * 100):.1f}%

Data Import:
- Total Records: {summary.total_records_imported:,}
- Duplicates Found: {summary.total_duplicates_found:,}
- Processing Time: {summary.total_processing_time_ms / 1000:.2f} seconds

File Management:
- Files Archived: {summary.files_archived}
- Archive Strategy: {self.movement_strategy.value}

Detailed Results:
"""
        
        for record in summary.ingestion_records:
            status_icon = {
                IngestionStatus.SUCCESS: "✓",
                IngestionStatus.WARNING: "⚠",
                IngestionStatus.ERROR: "✗",
                IngestionStatus.PENDING: "⏳"
            }.get(record.status, "?")
            
            report += f"\n{status_icon} {record.file_name}\n"
            report += f"   Status: {record.status.value}\n"
            report += f"   Records: {record.records_imported}/{record.records_extracted}\n"
            
            if record.duplicates_found > 0:
                report += f"   Duplicates: {record.duplicates_found}\n"
            
            if record.validation_errors > 0:
                report += f"   Validation Errors: {record.validation_errors}\n"
            
            if record.validation_warnings > 0:
                report += f"   Validation Warnings: {record.validation_warnings}\n"
            
            if record.archived:
                report += f"   Archived: Yes\n"
            
            if record.error_message:
                report += f"   Error: {record.error_message}\n"
        
        return report
    
    def cleanup_session(self) -> None:
        """Reset session tracking for new ingestion run."""
        self.current_session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.session_records = []
        logger.info(f"Started new ingestion session: {self.current_session_id}")