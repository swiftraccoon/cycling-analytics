"""
Data integrity validator for cycling analytics.

Validates data types, value ranges, and identifies anomalies without 
modifying the original data. Generates comprehensive validation reports.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, date
from pathlib import Path
import polars as pl
from pydantic import BaseModel, Field
from enum import Enum
import re

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


class ValidationIssue(BaseModel):
    """Individual validation issue."""
    
    column: str
    issue_type: str
    severity: ValidationSeverity
    message: str
    affected_rows: int
    sample_values: List[Any] = Field(default_factory=list)
    percentage: float = 0.0


class ColumnValidationResult(BaseModel):
    """Validation results for a single column."""
    
    column_name: str
    data_type: str
    expected_type: Optional[str] = None
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    min_value: Optional[Union[str, int, float]] = None
    max_value: Optional[Union[str, int, float]] = None
    issues: List[ValidationIssue] = Field(default_factory=list)


class ValidationReport(BaseModel):
    """Complete validation report for a dataset."""
    
    file_source: str
    file_hash: str
    validation_timestamp: datetime
    total_rows: int
    total_columns: int
    
    # Summary statistics
    critical_issues: int = 0
    error_issues: int = 0
    warning_issues: int = 0
    info_issues: int = 0
    
    # Column results
    column_results: List[ColumnValidationResult] = Field(default_factory=list)
    
    # Overall validation status
    validation_status: str = "success"  # success, warning, error, critical
    processing_time_ms: int = 0
    
    # Data quality score (0-100)
    quality_score: float = 100.0


class DataTypeValidator:
    """Validates data types and type consistency."""
    
    @staticmethod
    def validate_numeric(series: pl.Series) -> List[ValidationIssue]:
        """Validate numeric column."""
        issues = []
        
        # Check for non-numeric values (excluding nulls)
        non_null_series = series.drop_nulls()
        if len(non_null_series) == 0:
            return issues
        
        try:
            # Try to cast to numeric
            numeric_series = non_null_series.cast(pl.Float64, strict=False)
            null_after_cast = numeric_series.null_count()
            
            if null_after_cast > 0:
                issues.append(ValidationIssue(
                    column=series.name,
                    issue_type="invalid_numeric",
                    severity=ValidationSeverity.ERROR,
                    message=f"Found {null_after_cast} non-numeric values",
                    affected_rows=null_after_cast,
                    percentage=null_after_cast / len(non_null_series) * 100
                ))
                
        except Exception as e:
            issues.append(ValidationIssue(
                column=series.name,
                issue_type="type_validation_error",
                severity=ValidationSeverity.CRITICAL,
                message=f"Failed to validate numeric type: {e}",
                affected_rows=len(series)
            ))
        
        return issues
    
    @staticmethod
    def validate_boolean(series: pl.Series) -> List[ValidationIssue]:
        """Validate boolean column."""
        issues = []
        
        non_null_series = series.drop_nulls()
        if len(non_null_series) == 0:
            return issues
        
        # Check for valid boolean values
        valid_values = {'true', 'false', '1', '0', 'yes', 'no', 't', 'f'}
        string_series = non_null_series.cast(pl.Utf8).str.to_lowercase()
        
        invalid_mask = ~string_series.is_in(valid_values)
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            sample_invalid = string_series.filter(invalid_mask).unique().limit(5).to_list()
            
            issues.append(ValidationIssue(
                column=series.name,
                issue_type="invalid_boolean",
                severity=ValidationSeverity.ERROR,
                message=f"Found {invalid_count} invalid boolean values",
                affected_rows=invalid_count,
                sample_values=sample_invalid,
                percentage=invalid_count / len(non_null_series) * 100
            ))
        
        return issues
    
    @staticmethod
    def validate_datetime(series: pl.Series) -> List[ValidationIssue]:
        """Validate datetime column."""
        issues = []
        
        non_null_series = series.drop_nulls()
        if len(non_null_series) == 0:
            return issues
        
        # Common datetime patterns
        datetime_patterns = [
            r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO format
            r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',   # SQL format
            r'^\d{4}-\d{2}-\d{2}$',                     # Date only
            r'^\d{2}/\d{2}/\d{4}',                      # US format
        ]
        
        string_series = non_null_series.cast(pl.Utf8)
        valid_datetime_mask = pl.lit(False)
        
        for pattern in datetime_patterns:
            pattern_mask = string_series.str.contains(pattern)
            valid_datetime_mask = valid_datetime_mask | pattern_mask
        
        invalid_count = (~valid_datetime_mask).sum()
        
        if invalid_count > 0:
            sample_invalid = string_series.filter(~valid_datetime_mask).unique().limit(5).to_list()
            
            issues.append(ValidationIssue(
                column=series.name,
                issue_type="invalid_datetime_format",
                severity=ValidationSeverity.WARNING,
                message=f"Found {invalid_count} values with unexpected datetime format",
                affected_rows=invalid_count,
                sample_values=sample_invalid,
                percentage=invalid_count / len(non_null_series) * 100
            ))
        
        return issues


class ValueRangeValidator:
    """Validates value ranges and business rules."""
    
    # Expected ranges for cycling data
    CYCLING_RANGES = {
        'distance': {'min': 0, 'max': 1000000},  # 0 to 1000km in meters
        'moving_time': {'min': 0, 'max': 86400},  # 0 to 24 hours in seconds
        'elapsed_time': {'min': 0, 'max': 86400},  # 0 to 24 hours in seconds
        'total_elevation_gain': {'min': 0, 'max': 10000},  # 0 to 10,000m elevation
        'average_heartrate': {'min': 50, 'max': 220},  # Reasonable HR range
        'max_heartrate': {'min': 80, 'max': 250},  # Reasonable max HR
        'average_speed': {'min': 0, 'max': 30},  # 0 to 30 m/s (108 km/h)
        'max_speed': {'min': 0, 'max': 50},  # 0 to 50 m/s (180 km/h)
        'average_cadence': {'min': 30, 'max': 180},  # 30 to 180 rpm
        'max_cadence': {'min': 50, 'max': 220},  # 50 to 220 rpm
        'average_watts': {'min': 0, 'max': 2000},  # 0 to 2000 watts
        'max_watts': {'min': 0, 'max': 3000},  # 0 to 3000 watts
        'normalized_power': {'min': 0, 'max': 2000},  # 0 to 2000 watts
        'weighted_average_power': {'min': 0, 'max': 2000},  # 0 to 2000 watts
        'training_stress_score': {'min': 0, 'max': 1000},  # 0 to 1000 TSS
        'intensity_factor': {'min': 0, 'max': 2.0},  # 0 to 2.0 IF
        'calories': {'min': 0, 'max': 10000},  # 0 to 10,000 calories
    }
    
    def validate_range(
        self, 
        series: pl.Series, 
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> List[ValidationIssue]:
        """Validate numeric values are within specified range."""
        issues = []
        
        if series.dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
            return issues
        
        non_null_series = series.drop_nulls()
        if len(non_null_series) == 0:
            return issues
        
        # Check minimum value
        if min_val is not None:
            below_min = non_null_series < min_val
            below_min_count = below_min.sum()
            
            if below_min_count > 0:
                sample_values = non_null_series.filter(below_min).limit(5).to_list()
                
                issues.append(ValidationIssue(
                    column=series.name,
                    issue_type="value_below_minimum",
                    severity=ValidationSeverity.WARNING,
                    message=f"Found {below_min_count} values below minimum ({min_val})",
                    affected_rows=below_min_count,
                    sample_values=sample_values,
                    percentage=below_min_count / len(non_null_series) * 100
                ))
        
        # Check maximum value
        if max_val is not None:
            above_max = non_null_series > max_val
            above_max_count = above_max.sum()
            
            if above_max_count > 0:
                sample_values = non_null_series.filter(above_max).limit(5).to_list()
                
                issues.append(ValidationIssue(
                    column=series.name,
                    issue_type="value_above_maximum",
                    severity=ValidationSeverity.WARNING,
                    message=f"Found {above_max_count} values above maximum ({max_val})",
                    affected_rows=above_max_count,
                    sample_values=sample_values,
                    percentage=above_max_count / len(non_null_series) * 100
                ))
        
        return issues
    
    def validate_cycling_ranges(self, data: pl.DataFrame) -> List[ValidationIssue]:
        """Validate all cycling-specific ranges."""
        all_issues = []
        
        for column_name, range_config in self.CYCLING_RANGES.items():
            if column_name in data.columns:
                series = data[column_name]
                issues = self.validate_range(
                    series, 
                    range_config.get('min'),
                    range_config.get('max')
                )
                all_issues.extend(issues)
        
        return all_issues


class BusinessRuleValidator:
    """Validates business logic and data consistency."""
    
    def validate_time_consistency(self, data: pl.DataFrame) -> List[ValidationIssue]:
        """Validate time-related consistency rules."""
        issues = []
        
        # Moving time should not exceed elapsed time
        if 'moving_time' in data.columns and 'elapsed_time' in data.columns:
            inconsistent = data.filter(
                (data['moving_time'].is_not_null()) & 
                (data['elapsed_time'].is_not_null()) & 
                (data['moving_time'] > data['elapsed_time'])
            )
            
            if len(inconsistent) > 0:
                issues.append(ValidationIssue(
                    column="moving_time",
                    issue_type="time_inconsistency",
                    severity=ValidationSeverity.ERROR,
                    message=f"Moving time exceeds elapsed time in {len(inconsistent)} records",
                    affected_rows=len(inconsistent),
                    percentage=len(inconsistent) / len(data) * 100
                ))
        
        return issues
    
    def validate_power_consistency(self, data: pl.DataFrame) -> List[ValidationIssue]:
        """Validate power-related consistency rules."""
        issues = []
        
        # Average watts should not exceed max watts
        if 'average_watts' in data.columns and 'max_watts' in data.columns:
            inconsistent = data.filter(
                (data['average_watts'].is_not_null()) & 
                (data['max_watts'].is_not_null()) & 
                (data['average_watts'] > data['max_watts'])
            )
            
            if len(inconsistent) > 0:
                issues.append(ValidationIssue(
                    column="average_watts",
                    issue_type="power_inconsistency", 
                    severity=ValidationSeverity.WARNING,
                    message=f"Average watts exceeds max watts in {len(inconsistent)} records",
                    affected_rows=len(inconsistent),
                    percentage=len(inconsistent) / len(data) * 100
                ))
        
        return issues
    
    def validate_heartrate_consistency(self, data: pl.DataFrame) -> List[ValidationIssue]:
        """Validate heart rate consistency rules."""
        issues = []
        
        # Average HR should not exceed max HR
        if 'average_heartrate' in data.columns and 'max_heartrate' in data.columns:
            inconsistent = data.filter(
                (data['average_heartrate'].is_not_null()) & 
                (data['max_heartrate'].is_not_null()) & 
                (data['average_heartrate'] > data['max_heartrate'])
            )
            
            if len(inconsistent) > 0:
                issues.append(ValidationIssue(
                    column="average_heartrate",
                    issue_type="heartrate_inconsistency",
                    severity=ValidationSeverity.WARNING,
                    message=f"Average HR exceeds max HR in {len(inconsistent)} records",
                    affected_rows=len(inconsistent),
                    percentage=len(inconsistent) / len(data) * 100
                ))
        
        return issues


class IntegrityValidator:
    """
    Comprehensive data integrity validator for cycling analytics.
    
    Performs validation without modifying original data and generates
    detailed reports for data quality assessment.
    """
    
    def __init__(self):
        """Initialize validator with all validation components."""
        self.type_validator = DataTypeValidator()
        self.range_validator = ValueRangeValidator()
        self.business_validator = BusinessRuleValidator()
    
    def validate_column(
        self, 
        series: pl.Series,
        expected_type: Optional[str] = None
    ) -> ColumnValidationResult:
        """
        Validate a single column comprehensively.
        
        Args:
            series: Polars Series to validate
            expected_type: Expected data type
            
        Returns:
            Complete validation result for the column
        """
        column_name = series.name
        data_type = str(series.dtype)
        
        # Basic statistics
        null_count = series.null_count()
        null_percentage = null_count / len(series) * 100 if len(series) > 0 else 0
        unique_count = series.n_unique()
        unique_percentage = unique_count / len(series) * 100 if len(series) > 0 else 0
        
        # Min/max values (for applicable types)
        min_value = None
        max_value = None
        
        try:
            non_null_series = series.drop_nulls()
            if len(non_null_series) > 0:
                if series.dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                    min_value = float(non_null_series.min())
                    max_value = float(non_null_series.max())
                elif series.dtype == pl.Utf8:
                    # For strings, get length range
                    lengths = non_null_series.str.len_chars()
                    min_value = int(lengths.min())
                    max_value = int(lengths.max())
        except Exception as e:
            logger.warning(f"Failed to calculate min/max for {column_name}: {e}")
        
        # Collect validation issues
        issues = []
        
        # Type-specific validation
        if data_type.startswith('Float') or data_type.startswith('Int'):
            issues.extend(self.type_validator.validate_numeric(series))
        elif 'bool' in data_type.lower():
            issues.extend(self.type_validator.validate_boolean(series))
        elif 'date' in column_name.lower() or 'time' in column_name.lower():
            issues.extend(self.type_validator.validate_datetime(series))
        
        # Range validation for known cycling metrics
        if column_name.lower().replace(' ', '_') in self.range_validator.CYCLING_RANGES:
            range_config = self.range_validator.CYCLING_RANGES[column_name.lower().replace(' ', '_')]
            issues.extend(self.range_validator.validate_range(
                series, 
                range_config.get('min'),
                range_config.get('max')
            ))
        
        # High null percentage warning
        if null_percentage > 50:
            issues.append(ValidationIssue(
                column=column_name,
                issue_type="high_null_percentage",
                severity=ValidationSeverity.WARNING,
                message=f"High percentage of null values ({null_percentage:.1f}%)",
                affected_rows=null_count,
                percentage=null_percentage
            ))
        
        # Low uniqueness warning (except for boolean/categorical columns)
        if unique_percentage < 10 and series.dtype != pl.Boolean and len(series) > 100:
            issues.append(ValidationIssue(
                column=column_name,
                issue_type="low_uniqueness",
                severity=ValidationSeverity.INFO,
                message=f"Low data uniqueness ({unique_percentage:.1f}%)",
                affected_rows=len(series) - unique_count,
                percentage=100 - unique_percentage
            ))
        
        return ColumnValidationResult(
            column_name=column_name,
            data_type=data_type,
            expected_type=expected_type,
            null_count=null_count,
            null_percentage=null_percentage,
            unique_count=unique_count,
            unique_percentage=unique_percentage,
            min_value=min_value,
            max_value=max_value,
            issues=issues
        )
    
    def validate_dataset(
        self,
        data: pl.DataFrame,
        file_source: str,
        file_hash: str,
        expected_schema: Optional[Dict[str, str]] = None
    ) -> ValidationReport:
        """
        Validate entire dataset comprehensively.
        
        Args:
            data: DataFrame to validate
            file_source: Source file path
            file_hash: File hash for tracking
            expected_schema: Expected column types
            
        Returns:
            Complete validation report
        """
        start_time = datetime.utcnow()
        
        logger.info(f"Validating dataset: {len(data)} rows, {len(data.columns)} columns")
        
        # Initialize report
        report = ValidationReport(
            file_source=file_source,
            file_hash=file_hash,
            validation_timestamp=start_time,
            total_rows=len(data),
            total_columns=len(data.columns)
        )
        
        # Validate each column
        for column_name in data.columns:
            expected_type = expected_schema.get(column_name) if expected_schema else None
            column_result = self.validate_column(data[column_name], expected_type)
            report.column_results.append(column_result)
        
        # Business rule validation
        business_issues = []
        business_issues.extend(self.business_validator.validate_time_consistency(data))
        business_issues.extend(self.business_validator.validate_power_consistency(data))
        business_issues.extend(self.business_validator.validate_heartrate_consistency(data))
        
        # Add business rule issues to relevant columns
        for issue in business_issues:
            for col_result in report.column_results:
                if col_result.column_name == issue.column:
                    col_result.issues.append(issue)
                    break
        
        # Calculate issue counts and determine validation status
        all_issues = []
        for col_result in report.column_results:
            all_issues.extend(col_result.issues)
        
        for issue in all_issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                report.critical_issues += 1
            elif issue.severity == ValidationSeverity.ERROR:
                report.error_issues += 1
            elif issue.severity == ValidationSeverity.WARNING:
                report.warning_issues += 1
            else:
                report.info_issues += 1
        
        # Determine overall validation status
        if report.critical_issues > 0:
            report.validation_status = "critical"
        elif report.error_issues > 0:
            report.validation_status = "error"
        elif report.warning_issues > 0:
            report.validation_status = "warning"
        else:
            report.validation_status = "success"
        
        # Calculate data quality score
        total_issues = len(all_issues)
        total_possible_issues = len(data) * len(data.columns)  # Rough estimate
        
        if total_possible_issues > 0:
            issue_rate = total_issues / total_possible_issues
            report.quality_score = max(0, 100 - (issue_rate * 100))
        
        # Apply penalties for severe issues
        report.quality_score -= report.critical_issues * 10
        report.quality_score -= report.error_issues * 5
        report.quality_score -= report.warning_issues * 1
        report.quality_score = max(0, min(100, report.quality_score))
        
        # Calculate processing time
        processing_time = datetime.utcnow() - start_time
        report.processing_time_ms = int(processing_time.total_seconds() * 1000)
        
        logger.info(
            f"Validation complete: {report.validation_status} status, "
            f"quality score {report.quality_score:.1f}, "
            f"{total_issues} total issues"
        )
        
        return report
    
    def create_validation_summary(self, report: ValidationReport) -> str:
        """
        Create human-readable validation summary.
        
        Args:
            report: Validation report
            
        Returns:
            Formatted summary string
        """
        summary = f"""
Data Validation Report
=====================

Source: {Path(report.file_source).name}
Validated: {report.validation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Status: {report.validation_status.upper()}
Quality Score: {report.quality_score:.1f}/100

Dataset Overview:
- Total Rows: {report.total_rows:,}
- Total Columns: {report.total_columns}
- Processing Time: {report.processing_time_ms:,} ms

Issue Summary:
- Critical: {report.critical_issues}
- Errors: {report.error_issues}
- Warnings: {report.warning_issues}
- Info: {report.info_issues}

Column Analysis:
"""
        
        # Sort columns by issue count
        sorted_columns = sorted(
            report.column_results, 
            key=lambda x: len(x.issues), 
            reverse=True
        )
        
        for col in sorted_columns[:10]:  # Top 10 columns with issues
            if col.issues:
                summary += f"\n{col.column_name}:\n"
                summary += f"  - Type: {col.data_type}\n"
                summary += f"  - Null: {col.null_percentage:.1f}%\n"
                summary += f"  - Issues: {len(col.issues)}\n"
                
                for issue in col.issues[:3]:  # Top 3 issues per column
                    summary += f"    * {issue.severity.upper()}: {issue.message}\n"
        
        return summary