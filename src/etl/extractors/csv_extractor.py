"""CSV file extractor for cycling activity data."""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)


class CSVExtractor:
    """Extract cycling activity data from CSV files."""
    
    def __init__(self, bronze_path: str = "data/bronze"):
        """Initialize CSV extractor.
        
        Args:
            bronze_path: Path to bronze data directory
        """
        self.bronze_path = Path(bronze_path)
        self.incoming_path = self.bronze_path / "incoming"
        self.archive_path = self.bronze_path / "archive"
        
        # Ensure directories exist
        self.incoming_path.mkdir(parents=True, exist_ok=True)
        self.archive_path.mkdir(parents=True, exist_ok=True)
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex digest of file hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def extract_csv(self, file_path: Path) -> tuple[pl.DataFrame, dict]:
        """Extract data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        logger.info(f"Extracting data from {file_path}")
        
        # Create metadata
        metadata = {
            "file_path": str(file_path),
            "file_name": file_path.name if hasattr(file_path, 'name') else str(file_path),
            "file_hash": None,
            "file_size": 0,
            "extraction_timestamp": datetime.now(),
            "status": "pending",
            "record_count": 0,
        }
        
        # Check if file exists
        if not file_path.exists():
            metadata["status"] = "error"
            metadata["error"] = f"File not found: {file_path}"
            return pl.DataFrame(), metadata
        
        # Calculate file hash and size
        try:
            file_hash = self.calculate_file_hash(file_path)
            metadata["file_hash"] = file_hash
            metadata["file_size"] = file_path.stat().st_size
        except Exception as e:
            logger.warning(f"Could not calculate hash for {file_path}: {e}")
        
        # Read CSV with Polars - no modifications to data
        try:
            df = pl.read_csv(
                file_path,
                try_parse_dates=True,
                null_values=["", "NA", "null", "None"],
                ignore_errors=False,  # Fail on parsing errors
            )
            
            # Add metadata columns to track source
            df = df.with_columns([
                pl.lit(file_path.name).alias("file_source"),
                pl.lit(file_hash).alias("file_hash"),
                pl.lit(datetime.now()).alias("import_timestamp"),
            ])
            
            metadata["record_count"] = len(df)
            metadata["columns"] = df.columns
            metadata["status"] = "success"
            
            logger.info(f"Extracted {len(df)} records from {file_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to extract CSV {file_path}: {e}")
            metadata["status"] = "error"
            metadata["error"] = str(e)
            return pl.DataFrame(), metadata
        
        return df, metadata
    
    def extract_from_file(self, file_path: Path) -> tuple[pl.DataFrame, dict]:
        """Extract data from a single CSV file (alias for extract_csv).
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        return self.extract_csv(file_path)
    
    def get_incoming_files(self) -> list[Path]:
        """Get list of CSV files in incoming directory.
        
        Returns:
            List of CSV file paths
        """
        csv_files = sorted(self.incoming_path.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files in incoming directory")
        return csv_files
    
    def get_archive_files(self) -> list[Path]:
        """Get list of CSV files in archive directory.
        
        Returns:
            List of CSV file paths
        """
        csv_files = sorted(self.archive_path.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files in archive directory")
        return csv_files
    
    def extract_all(self, include_archive: bool = True) -> tuple[pl.DataFrame, list[dict]]:
        """Extract data from all available CSV files.
        
        Args:
            include_archive: Whether to include archive files
            
        Returns:
            Tuple of (combined DataFrame, list of metadata dicts)
        """
        all_data = []
        all_metadata = []
        
        # Process incoming files
        for file_path in self.get_incoming_files():
            try:
                df, metadata = self.extract_csv(file_path)
                all_data.append(df)
                all_metadata.append(metadata)
            except Exception as e:
                logger.error(f"Skipping file {file_path} due to error: {e}")
        
        # Process archive files if requested
        if include_archive:
            for file_path in self.get_archive_files():
                try:
                    df, metadata = self.extract_csv(file_path)
                    all_data.append(df)
                    all_metadata.append(metadata)
                except Exception as e:
                    logger.error(f"Skipping file {file_path} due to error: {e}")
        
        # Combine all dataframes
        if all_data:
            combined_df = pl.concat(all_data, how="diagonal")  # Handle schema differences
            logger.info(f"Combined {len(all_data)} files with {len(combined_df)} total records")
        else:
            combined_df = pl.DataFrame()
            logger.warning("No data extracted from any files")
        
        return combined_df, all_metadata
    
    def move_to_archive(self, file_path: Path) -> Path:
        """Move processed file from incoming to archive.
        
        Args:
            file_path: Path to file in incoming directory
            
        Returns:
            New path in archive directory
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Generate unique name if file already exists in archive
        archive_path = self.archive_path / file_path.name
        if archive_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = file_path.stem
            suffix = file_path.suffix
            archive_path = self.archive_path / f"{stem}_{timestamp}{suffix}"
        
        # Move file
        file_path.rename(archive_path)
        logger.info(f"Moved {file_path.name} to archive as {archive_path.name}")
        
        return archive_path