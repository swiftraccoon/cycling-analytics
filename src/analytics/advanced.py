"""Advanced analytics for cycling performance."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy import signal
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class AdvancedAnalytics:
    """Advanced cycling analytics including power curves and seasonal analysis."""
    
    def __init__(self, activities_df: pl.DataFrame):
        """Initialize advanced analytics.
        
        Args:
            activities_df: DataFrame with activity data
        """
        self.df = activities_df
    
    def calculate_power_curve(
        self, 
        durations: Optional[List[int]] = None,
        percentile: int = 95
    ) -> pl.DataFrame:
        """Calculate power curve for specified durations.
        
        Args:
            durations: List of durations in seconds (default: standard durations)
            percentile: Percentile to use for best efforts (default: 95th)
            
        Returns:
            DataFrame with duration and power values
        """
        if durations is None:
            # Standard durations: 5s, 10s, 30s, 1m, 2m, 5m, 10m, 20m, 60m
            durations = [5, 10, 30, 60, 120, 300, 600, 1200, 3600]
        
        power_curve_data = []
        
        # Check if required column exists
        if "moving_time" not in self.df.columns:
            logger.warning("No moving_time column found for power curve")
            return pl.DataFrame({"duration": [], "power": []})
        
        for duration in durations:
            # Get activities with sufficient duration
            valid_activities = self.df.filter(
                pl.col("moving_time") >= duration
            )
            
            if len(valid_activities) == 0:
                continue
            
            # Calculate average power for the duration
            # Using normalized power as proxy for now
            if "icu_normalized_watts" in valid_activities.columns:
                powers = valid_activities["icu_normalized_watts"].drop_nulls()
                if len(powers) > 0:
                    # Get percentile value
                    power_value = np.percentile(powers, percentile)
                    power_curve_data.append({
                        "duration_seconds": duration,
                        "duration_minutes": duration / 60,
                        "power": power_value,
                        "sample_size": len(powers),
                    })
        
        if not power_curve_data:
            return pl.DataFrame()
        
        return pl.DataFrame(power_curve_data)
    
    def calculate_seasonal_patterns(self) -> Dict:
        """Analyze seasonal training patterns.
        
        Returns:
            Dictionary with seasonal analysis
        """
        # Check if required column exists
        if "start_date_local" not in self.df.columns:
            logger.warning("No start_date_local column found for seasonal patterns")
            return {}
        
        # Convert dates if needed
        if self.df["start_date_local"].dtype == pl.Utf8:
            df = self.df.with_columns(
                pl.col("start_date_local").str.to_datetime()
            )
        else:
            df = self.df
        
        # Add temporal columns
        df = df.with_columns([
            pl.col("start_date_local").dt.month().alias("month"),
            pl.col("start_date_local").dt.quarter().alias("quarter"),
            pl.col("start_date_local").dt.year().alias("year"),
            pl.col("start_date_local").dt.week().alias("week"),
        ])
        
        # Monthly patterns with conditional aggregations
        monthly_aggs = [pl.count().alias("activity_count")]
        if "moving_time" in df.columns:
            monthly_aggs.append(pl.col("moving_time").sum().alias("total_time"))
        if "distance" in df.columns:
            monthly_aggs.append(pl.col("distance").sum().alias("total_distance"))
        if "icu_training_load" in df.columns:
            monthly_aggs.append(pl.col("icu_training_load").sum().alias("total_load"))
        if "icu_average_watts" in df.columns:
            monthly_aggs.append(pl.col("icu_average_watts").mean().alias("avg_power"))
        
        monthly_stats = df.group_by("month").agg(monthly_aggs).sort("month")
        
        # Quarterly patterns with conditional aggregations
        quarterly_aggs = [pl.count().alias("activity_count")]
        if "moving_time" in df.columns:
            quarterly_aggs.append(pl.col("moving_time").sum().alias("total_time"))
        if "distance" in df.columns:
            quarterly_aggs.append(pl.col("distance").sum().alias("total_distance"))
        if "icu_training_load" in df.columns:
            quarterly_aggs.append(pl.col("icu_training_load").sum().alias("total_load"))
        
        quarterly_stats = df.group_by(["year", "quarter"]).agg(quarterly_aggs).sort(["year", "quarter"])
        
        # Day of week patterns with conditional aggregations
        dow_aggs = [pl.count().alias("activity_count")]
        if "moving_time" in df.columns:
            dow_aggs.append(pl.col("moving_time").mean().alias("avg_duration"))
        if "icu_training_load" in df.columns:
            dow_aggs.append(pl.col("icu_training_load").mean().alias("avg_load"))
        
        dow_stats = df.with_columns(
            pl.col("start_date_local").dt.weekday().alias("day_of_week")
        ).group_by("day_of_week").agg(dow_aggs).sort("day_of_week")
        
        # Time of day patterns with conditional aggregations
        hour_aggs = [pl.count().alias("activity_count")]
        if "icu_average_watts" in df.columns:
            hour_aggs.append(pl.col("icu_average_watts").mean().alias("avg_power"))
        
        hour_stats = df.with_columns(
            pl.col("start_date_local").dt.hour().alias("hour")
        ).group_by("hour").agg(hour_aggs).sort("hour")
        
        return {
            "monthly": monthly_stats,
            "quarterly": quarterly_stats,
            "day_of_week": dow_stats,
            "time_of_day": hour_stats,
        }
    
    def calculate_variability_index(self, window_days: int = 7) -> pl.DataFrame:
        """Calculate Variability Index (VI) for training consistency.
        
        Args:
            window_days: Rolling window size in days
            
        Returns:
            DataFrame with VI values over time
        """
        # Convert dates if needed
        if self.df["start_date_local"].dtype == pl.Utf8:
            df = self.df.with_columns(
                pl.col("start_date_local").str.to_datetime()
            )
        else:
            df = self.df
        
        # Group by date and sum training load
        daily_load = df.group_by(
            pl.col("start_date_local").dt.date().alias("date")
        ).agg(
            pl.col("icu_training_load").sum().alias("daily_load")
        ).sort("date")
        
        # Fill missing dates with zero load
        date_range = pl.date_range(
            daily_load["date"].min(),
            daily_load["date"].max(),
            interval="1d",
            eager=True,
        )
        
        full_dates = pl.DataFrame({"date": date_range})
        daily_load = full_dates.join(daily_load, on="date", how="left").fill_null(0)
        
        # Calculate rolling statistics
        daily_load = daily_load.with_columns([
            pl.col("daily_load").rolling_mean(window_days).alias("rolling_mean"),
            pl.col("daily_load").rolling_std(window_days).alias("rolling_std"),
        ])
        
        # Calculate Variability Index (CV = std/mean)
        daily_load = daily_load.with_columns(
            (pl.col("rolling_std") / pl.col("rolling_mean")).alias("variability_index")
        )
        
        return daily_load
    
    def detect_training_phases(self) -> Dict:
        """Detect training phases (base, build, peak, recovery).
        
        Returns:
            Dictionary with detected training phases
        """
        # Convert dates if needed
        if self.df["start_date_local"].dtype == pl.Utf8:
            df = self.df.with_columns(
                pl.col("start_date_local").str.to_datetime()
            )
        else:
            df = self.df
        
        # Calculate weekly metrics
        weekly_data = df.with_columns(
            pl.col("start_date_local").dt.truncate("1w").alias("week")
        ).group_by("week").agg([
            pl.col("icu_training_load").sum().alias("weekly_load"),
            pl.col("moving_time").sum().alias("weekly_time"),
            pl.col("icu_intensity").mean().alias("avg_intensity"),
            pl.count().alias("activity_count"),
        ]).sort("week")
        
        if weekly_data.is_empty():
            return {}
        
        # Calculate rolling averages
        weekly_data = weekly_data.with_columns([
            pl.col("weekly_load").rolling_mean(4).alias("load_trend"),
            pl.col("avg_intensity").rolling_mean(4).alias("intensity_trend"),
        ])
        
        # Detect phases based on load and intensity patterns
        phases = []
        
        for i in range(len(weekly_data)):
            row = weekly_data[i]
            load = row["weekly_load"][0]
            intensity = row["avg_intensity"][0] if row["avg_intensity"][0] is not None else 0
            load_trend = row["load_trend"][0] if row["load_trend"][0] is not None else load
            
            # Simple phase detection logic
            if load < load_trend * 0.7:
                phase = "Recovery"
            elif intensity > 0.85 and load > load_trend * 0.9:
                phase = "Peak/Race"
            elif load > load_trend * 1.1:
                phase = "Build"
            else:
                phase = "Base/Maintenance"
            
            phases.append({
                "week": row["week"][0],
                "phase": phase,
                "weekly_load": load,
                "avg_intensity": intensity,
            })
        
        return {
            "phases": pl.DataFrame(phases),
            "current_phase": phases[-1]["phase"] if phases else "Unknown",
        }
    
    def calculate_chronic_training_balance(self, days: int = 90) -> Dict:
        """Calculate chronic training balance metrics.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with balance metrics
        """
        # Convert dates if needed
        if self.df["start_date_local"].dtype == pl.Utf8:
            df = self.df.with_columns(
                pl.col("start_date_local").str.to_datetime()
            )
        else:
            df = self.df
        
        end_date = df["start_date_local"].max()
        start_date = end_date - timedelta(days=days)
        
        recent_df = df.filter(pl.col("start_date_local") >= start_date)
        
        if recent_df.is_empty():
            return {}
        
        # Calculate zone balance
        zone_balance = {}
        for i in range(1, 8):
            col_name = f"z{i}_secs"
            if col_name in recent_df.columns:
                total_secs = recent_df[col_name].sum()
                zone_balance[f"Z{i}"] = total_secs / 3600  # Convert to hours
        
        total_zone_time = sum(zone_balance.values())
        
        # Calculate zone percentages
        zone_percentages = {}
        if total_zone_time > 0:
            for zone, hours in zone_balance.items():
                zone_percentages[zone] = (hours / total_zone_time) * 100
        
        # Calculate intensity distribution
        easy_zones = zone_percentages.get("Z1", 0) + zone_percentages.get("Z2", 0)
        moderate_zones = zone_percentages.get("Z3", 0) + zone_percentages.get("Z4", 0)
        hard_zones = zone_percentages.get("Z5", 0) + zone_percentages.get("Z6", 0) + zone_percentages.get("Z7", 0)
        
        # Polarization index (ratio of easy+hard to moderate)
        if moderate_zones > 0:
            polarization_index = (easy_zones + hard_zones) / moderate_zones
        else:
            polarization_index = None
        
        return {
            "zone_hours": zone_balance,
            "zone_percentages": zone_percentages,
            "intensity_distribution": {
                "easy": easy_zones,
                "moderate": moderate_zones,
                "hard": hard_zones,
            },
            "polarization_index": polarization_index,
            "total_hours": total_zone_time,
        }
    
    def estimate_vo2max_progression(self) -> pl.DataFrame:
        """Estimate VO2max progression from power data.
        
        Returns:
            DataFrame with estimated VO2max values over time
        """
        # Filter activities with power data
        power_df = self.df.filter(
            pl.col("icu_normalized_watts").is_not_null()
        )
        
        if power_df.is_empty():
            return pl.DataFrame()
        
        # Convert dates if needed
        if power_df["start_date_local"].dtype == pl.Utf8:
            power_df = power_df.with_columns(
                pl.col("start_date_local").str.to_datetime()
            )
        
        # Group by month and calculate metrics
        monthly_vo2 = power_df.with_columns(
            pl.col("start_date_local").dt.truncate("1mo").alias("month")
        ).group_by("month").agg([
            # Use normalized power as proxy for VO2max estimation
            # This is a simplified estimation
            pl.col("icu_normalized_watts").max().alias("max_np"),
            pl.col("icu_normalized_watts").quantile(0.95).alias("p95_np"),
            pl.col("icu_average_watts").mean().alias("avg_power"),
        ]).sort("month")
        
        # Add estimated VO2max (simplified formula)
        # Real VO2max calculation would require weight and more sophisticated modeling
        monthly_vo2 = monthly_vo2.with_columns(
            # Rough estimation: VO2max ≈ 10.8 * (W/kg) + 7
            # Assuming 70kg rider for now
            ((pl.col("p95_np") / 70) * 10.8 + 7).alias("estimated_vo2max")
        )
        
        return monthly_vo2