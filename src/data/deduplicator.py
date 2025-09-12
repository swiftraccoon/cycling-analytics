"""Duplicate detection and resolution for cycling activities."""

import logging
from datetime import datetime
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)


class Deduplicator:
    """Handle duplicate detection and resolution for activity data."""
    
    def __init__(self):
        """Initialize deduplicator."""
        self.deduplication_report = {
            "total_records": 0,
            "unique_records": 0,
            "exact_duplicates": 0,
            "potential_duplicates": 0,
            "resolution_strategy": "keep_latest",
            "timestamp": None,
        }
    
    def identify_exact_duplicates(self, df: pl.DataFrame) -> pl.DataFrame:
        """Identify exact duplicates by activity ID.
        
        Args:
            df: DataFrame with activity data
            
        Returns:
            DataFrame with duplicate flags added
        """
        if "id" not in df.columns:
            logger.warning("No 'id' column found, cannot identify exact duplicates")
            return df
        
        # Count occurrences of each ID
        if "import_timestamp" in df.columns:
            id_counts = df.group_by("id").agg(
                pl.len().alias("duplicate_count"),
                pl.col("import_timestamp").min().alias("first_import"),
                pl.col("import_timestamp").max().alias("last_import"),
            )
        else:
            id_counts = df.group_by("id").agg(
                pl.len().alias("duplicate_count"),
            )
        
        # Join back to mark duplicates
        df = df.join(id_counts, on="id", how="left")
        
        # Flag exact duplicates
        df = df.with_columns(
            (pl.col("duplicate_count") > 1).alias("is_exact_duplicate")
        )
        
        exact_duplicates = df.filter(pl.col("is_exact_duplicate")).height
        logger.info(f"Found {exact_duplicates} exact duplicate records")
        
        return df
    
    def identify_potential_duplicates(self, df: pl.DataFrame) -> pl.DataFrame:
        """Identify potential duplicates by timestamp and similar metrics.
        
        Args:
            df: DataFrame with activity data
            
        Returns:
            DataFrame with potential duplicate flags added
        """
        required_cols = ["start_date_local", "distance", "moving_time"]
        if not all(col in df.columns for col in required_cols):
            logger.warning("Missing required columns for potential duplicate detection")
            return df
        
        # Create a hash of key fields for comparison
        hash_cols = [pl.col("start_date_local").cast(pl.Utf8)]
        
        if "type" in df.columns:
            hash_cols.append(pl.col("type").fill_null("unknown"))
        
        hash_cols.extend([
            pl.col("distance").round(0).cast(pl.Utf8).fill_null("0"),
            pl.col("moving_time").round(0).cast(pl.Utf8).fill_null("0"),
        ])
        
        df = df.with_columns(
            pl.concat_str(hash_cols, separator="_").alias("activity_hash")
        )
        
        # Count occurrences of each hash
        hash_counts = df.group_by("activity_hash").agg(
            pl.len().alias("potential_duplicate_count")
        )
        
        # Join back to mark potential duplicates
        df = df.join(hash_counts, on="activity_hash", how="left")
        
        # Flag potential duplicates (but not exact duplicates)
        df = df.with_columns(
            ((pl.col("potential_duplicate_count") > 1) & 
             ~pl.col("is_exact_duplicate").fill_null(False)).alias("is_potential_duplicate")
        )
        
        potential_duplicates = df.filter(pl.col("is_potential_duplicate")).height
        logger.info(f"Found {potential_duplicates} potential duplicate records")
        
        return df
    
    def resolve_duplicates(
        self, 
        df: pl.DataFrame, 
        strategy: str = "keep_latest",
        preserve_all: bool = False
    ) -> pl.DataFrame:
        """Resolve duplicates according to specified strategy.
        
        Args:
            df: DataFrame with duplicate flags
            strategy: Resolution strategy ('keep_latest', 'keep_first', 'keep_all')
            preserve_all: If True, keep all records but flag resolution
            
        Returns:
            DataFrame with duplicates resolved
        """
        original_count = len(df)
        
        if strategy == "keep_latest" and "import_timestamp" in df.columns:
            # Sort by import timestamp descending and keep first occurrence of each ID
            df = df.sort("import_timestamp", descending=True)
            
            if not preserve_all and "id" in df.columns:
                # Keep only the latest version of each activity
                df_resolved = df.unique(subset=["id"], keep="first")
            else:
                # Mark which records would be kept
                df_resolved = df.with_columns(
                    pl.col("id").is_first_distinct().alias("would_keep")
                )
        
        elif strategy == "keep_first" and "import_timestamp" in df.columns:
            # Sort by import timestamp ascending and keep first occurrence
            df = df.sort("import_timestamp")
            
            if not preserve_all and "id" in df.columns:
                df_resolved = df.unique(subset=["id"], keep="first")
            else:
                df_resolved = df.with_columns(
                    pl.col("id").is_first_distinct().alias("would_keep")
                )
        
        else:
            # Keep all records
            df_resolved = df
            logger.info("Keeping all records (no deduplication applied)")
        
        resolved_count = len(df_resolved)
        removed_count = original_count - resolved_count
        
        # Update report
        self.deduplication_report.update({
            "total_records": original_count,
            "unique_records": resolved_count,
            "exact_duplicates": removed_count,
            "resolution_strategy": strategy,
            "timestamp": datetime.now(),
        })
        
        logger.info(f"Resolved duplicates: {original_count} -> {resolved_count} records")
        
        return df_resolved
    
    def deduplicate(
        self, 
        df: pl.DataFrame,
        strategy: str = "keep_latest",
        identify_potential: bool = True,
        preserve_all: bool = False
    ) -> tuple[pl.DataFrame, dict]:
        """Complete deduplication pipeline.
        
        Args:
            df: Input DataFrame
            strategy: Resolution strategy
            identify_potential: Whether to identify potential duplicates
            preserve_all: If True, keep all records but flag them
            
        Returns:
            Tuple of (deduplicated DataFrame, deduplication report)
        """
        logger.info(f"Starting deduplication of {len(df)} records")
        start_time = datetime.now()
        
        # Initialize report
        self.deduplication_report["total_records"] = len(df)
        
        # Step 1: Identify exact duplicates
        df = self.identify_exact_duplicates(df)
        
        # Count exact duplicates
        exact_duplicates = 0
        if "is_exact_duplicate" in df.columns:
            # Count records that are duplicates (not unique IDs)
            exact_duplicates = df.filter(pl.col("is_exact_duplicate")).height
            # But we only count removed duplicates
            unique_ids = df["id"].n_unique() if "id" in df.columns else len(df)
            exact_duplicates = len(df) - unique_ids
        
        # Step 2: Identify potential duplicates if requested
        potential_duplicates = 0
        if identify_potential:
            df = self.identify_potential_duplicates(df)
            if "is_potential_duplicate" in df.columns:
                potential_duplicates = df.filter(pl.col("is_potential_duplicate")).height
        
        # Step 3: Resolve duplicates
        df = self.resolve_duplicates(df, strategy=strategy, preserve_all=preserve_all)
        
        # Step 4: Add lineage information
        if "file_source" in df.columns and "id" in df.columns:
            file_sources = df.group_by("id").agg(
                pl.col("file_source").unique().alias("all_file_sources_list")
            )
            # Convert list to string
            file_sources = file_sources.with_columns(
                pl.col("all_file_sources_list").list.join("; ").alias("all_file_sources")
            ).drop("all_file_sources_list")
            df = df.join(file_sources, on="id", how="left")
        
        # Update report
        self.deduplication_report["unique_records"] = len(df)
        self.deduplication_report["exact_duplicates"] = exact_duplicates
        self.deduplication_report["potential_duplicates"] = potential_duplicates
        self.deduplication_report["resolution_strategy"] = strategy
        self.deduplication_report["timestamp"] = datetime.now()
        self.deduplication_report["deduplication_time"] = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Deduplication complete: {self.deduplication_report}")
        
        return df, self.deduplication_report
    
    def get_duplicate_summary(self, df: pl.DataFrame) -> dict:
        """Get summary statistics about duplicates in the data.
        
        Args:
            df: DataFrame with duplicate flags
            
        Returns:
            Dictionary with duplicate statistics
        """
        summary = {
            "total_records": len(df),
            "unique_ids": df["id"].n_unique() if "id" in df.columns else 0,
            "exact_duplicates": 0,
            "potential_duplicates": 0,
            "files_with_duplicates": [],
        }
        
        if "is_exact_duplicate" in df.columns:
            summary["exact_duplicates"] = df.filter(pl.col("is_exact_duplicate")).height
        
        if "is_potential_duplicate" in df.columns:
            summary["potential_duplicates"] = df.filter(pl.col("is_potential_duplicate")).height
        
        if "file_source" in df.columns and "is_exact_duplicate" in df.columns:
            files_with_dups = (
                df.filter(pl.col("is_exact_duplicate"))
                .select("file_source")
                .unique()
                .to_series()
                .to_list()
            )
            summary["files_with_duplicates"] = files_with_dups
        
        return summary
    
    def track_file_sources(self, df: pl.DataFrame) -> pl.DataFrame:
        """Track all file sources for each activity.
        
        Args:
            df: DataFrame with activity data
            
        Returns:
            DataFrame with consolidated file sources
        """
        if "id" not in df.columns or "file_source" not in df.columns:
            logger.warning("Missing required columns for file source tracking")
            return df
        
        # Group by ID and aggregate file sources
        file_sources = df.group_by("id").agg(
            pl.col("file_source").unique().sort().str.join(", ").alias("all_file_sources")
        )
        
        # Join back to original dataframe - get first row per ID
        df_first = df.unique(subset=["id"], keep="first")
        df_with_sources = df_first.drop("all_file_sources") if "all_file_sources" in df_first.columns else df_first
        df_with_sources = df_with_sources.join(file_sources, on="id", how="left")
        
        return df_with_sources