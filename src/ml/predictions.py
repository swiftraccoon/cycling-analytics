"""Machine learning predictions for cycling performance."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PerformancePredictor:
    """Predict cycling performance metrics using ML models."""
    
    def __init__(self, activities_df: pl.DataFrame):
        """Initialize performance predictor.
        
        Args:
            activities_df: DataFrame with activity data
        """
        self.df = activities_df
        self.models = {}
        self.scalers = {}
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, pl.DataFrame]:
        """Prepare data for training models.
        
        Returns:
            Tuple of (features, target, processed_df)
        """
        # Convert dates if needed
        if self.df["start_date_local"].dtype == pl.Utf8:
            df = self.df.with_columns(
                pl.col("start_date_local").str.to_datetime()
            )
        else:
            df = self.df
        
        # Create unified FTP column using safe approach
        ftp_expr = pl.lit(None)
        
        if "icu_ftp" in df.columns:
            ftp_expr = pl.when(pl.col("icu_ftp").is_not_null()).then(pl.col("icu_ftp")).otherwise(ftp_expr)
        
        if "threshold_power" in df.columns:
            ftp_expr = pl.when(pl.col("threshold_power").is_not_null()).then(pl.col("threshold_power")).otherwise(ftp_expr)
        
        if "functional_threshold_power" in df.columns:
            ftp_expr = pl.when(pl.col("functional_threshold_power").is_not_null()).then(pl.col("functional_threshold_power")).otherwise(ftp_expr)
        
        df_with_ftp = df.with_columns(ftp_expr.alias("ftp_value"))
        
        ftp_df = df_with_ftp.filter(pl.col("ftp_value").is_not_null()).sort("start_date_local")
        
        if len(ftp_df) < 10:
            logger.warning("Insufficient data for FTP prediction (need at least 10 data points)")
            return None, None, pl.DataFrame()
        
        # Calculate rolling features based on available columns
        rolling_features = [
            # Days since first activity
            ((pl.col("start_date_local") - pl.col("start_date_local").min()).dt.total_days()).alias("days_training"),
        ]
        
        feature_cols = ["days_training"]
        
        # Add features only if columns exist
        if "icu_training_load" in ftp_df.columns:
            rolling_features.extend([
                pl.col("icu_training_load").rolling_mean(7).alias("load_7d"),
                pl.col("icu_training_load").rolling_mean(28).alias("load_28d"),
            ])
            feature_cols.extend(["load_7d", "load_28d"])
        
        if "moving_time" in ftp_df.columns:
            rolling_features.extend([
                pl.col("moving_time").rolling_sum(7).alias("time_7d"),
                pl.col("moving_time").rolling_sum(28).alias("time_28d"),
            ])
            feature_cols.extend(["time_7d", "time_28d"])
        
        if "icu_intensity" in ftp_df.columns:
            rolling_features.append(pl.col("icu_intensity").rolling_mean(7).alias("intensity_7d"))
            feature_cols.append("intensity_7d")
        
        if "icu_fitness" in ftp_df.columns:
            feature_cols.append("icu_fitness")
        
        if "icu_fatigue" in ftp_df.columns:
            feature_cols.append("icu_fatigue")
        
        ftp_df = ftp_df.with_columns(rolling_features)
        
        # Filter feature_cols to only those that exist after transformation
        existing_feature_cols = [col for col in feature_cols if col in ftp_df.columns]
        
        if not existing_feature_cols:
            logger.warning("No features available for training")
            return None, None, pl.DataFrame()
        
        # Drop rows with null features
        ftp_df = ftp_df.drop_nulls(subset=existing_feature_cols + ["ftp_value"])
        
        if len(ftp_df) < 5:
            logger.warning("Insufficient clean data for prediction")
            return None, None, pl.DataFrame()
        
        # Extract features and target
        X = ftp_df.select(existing_feature_cols).to_numpy()
        y = ftp_df["ftp_value"].to_numpy()
        
        return X, y, ftp_df
    
    def predict_ftp_progression(self, days_ahead: int = 30) -> Dict:
        """Predict FTP progression.
        
        Args:
            days_ahead: Number of days to predict ahead
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        # Check if we have any data first
        # Check if we have any FTP data from any source
        has_ftp = False
        if "icu_ftp" in self.df.columns and self.df["icu_ftp"].is_not_null().any():
            has_ftp = True
        elif "threshold_power" in self.df.columns and self.df["threshold_power"].is_not_null().any():
            has_ftp = True
        elif "functional_threshold_power" in self.df.columns and self.df["functional_threshold_power"].is_not_null().any():
            has_ftp = True
        
        if self.df.is_empty() or not has_ftp:
            return {
                "error": "No FTP data available",
                "message": "Need activities with FTP data for prediction",
                "current_activities": len(self.df),
            }
        
        # Get unique FTP values to check progression
        # Use the same unified FTP approach as PerformanceAnalyzer
        ftp_expr = pl.lit(None)
        
        if "icu_ftp" in self.df.columns:
            ftp_expr = pl.when(pl.col("icu_ftp").is_not_null()).then(pl.col("icu_ftp")).otherwise(ftp_expr)
        
        if "threshold_power" in self.df.columns:
            ftp_expr = pl.when(pl.col("threshold_power").is_not_null()).then(pl.col("threshold_power")).otherwise(ftp_expr)
        
        if "functional_threshold_power" in self.df.columns:
            ftp_expr = pl.when(pl.col("functional_threshold_power").is_not_null()).then(pl.col("functional_threshold_power")).otherwise(ftp_expr)
        
        df_with_ftp = self.df.with_columns(ftp_expr.alias("ftp_value"))
        
        ftp_data = df_with_ftp.filter(pl.col("ftp_value").is_not_null())
        unique_ftps = ftp_data["ftp_value"].unique()
        
        # Need at least 3 different FTP values for meaningful prediction
        if len(unique_ftps) < 3:
            return {
                "error": "Insufficient FTP progression data",
                "message": f"Only {len(unique_ftps)} FTP values found. Need more FTP tests for prediction.",
                "current_ftp_values": len(unique_ftps),
                "required_ftp_values": 3,
            }
        
        X, y, processed_df = self.prepare_training_data()
        
        if X is None or len(X) < 3:
            return {
                "error": "Insufficient training data",
                "message": "Need at least 3 activities with FTP data",
                "current_activities": len(X) if X is not None else 0,
            }
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers["ftp"] = scaler
        
        # Train multiple models for ensemble
        models = {
            "rf": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5),
            "gb": GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3),
            "lr": LinearRegression(),
        }
        
        # Time series cross-validation - adjust splits for small datasets
        n_splits = min(2, max(1, len(X) // 2))  # At least 1 split, max 2 for small data
        
        model_scores = {}
        
        # Only use simple models for small datasets
        if len(X) < 10:
            # For very small datasets, just use linear regression
            models = {"lr": LinearRegression()}
        
        for name, model in models.items():
            if n_splits > 1:
                tscv = TimeSeriesSplit(n_splits=n_splits)
                try:
                    scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2')
                    model_scores[name] = scores.mean()
                except:
                    model_scores[name] = 0
            else:
                model_scores[name] = 0
            
            model.fit(X_scaled, y)
            self.models[f"ftp_{name}"] = model
        
        # Make predictions
        current_ftp = y[-1]
        last_features = X[-1]
        
        # Simulate future features (simplified)
        future_dates = []
        future_predictions = []
        confidence_lower = []
        confidence_upper = []
        
        for day in range(1, days_ahead + 1):
            # Update features for future prediction
            future_features = last_features.copy()
            future_features[0] += day  # Increment days_training
            
            # Make ensemble prediction
            future_scaled = scaler.transform(future_features.reshape(1, -1))
            predictions = []
            for name, model in models.items():
                pred = model.predict(future_scaled)[0]
                predictions.append(pred)
            
            # Calculate mean and confidence interval
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            
            future_predictions.append(mean_pred)
            confidence_lower.append(mean_pred - 1.96 * std_pred)
            confidence_upper.append(mean_pred + 1.96 * std_pred)
            
            future_date = processed_df["start_date_local"].max() + timedelta(days=day)
            future_dates.append(future_date)
        
        return {
            "current_ftp": current_ftp,
            "predictions": {
                "dates": future_dates,
                "ftp_values": future_predictions,
                "confidence_lower": confidence_lower,
                "confidence_upper": confidence_upper,
            },
            "model_scores": model_scores,
            "best_model": max(model_scores, key=model_scores.get),
            "expected_gain": future_predictions[-1] - current_ftp,
            "expected_gain_percentage": ((future_predictions[-1] - current_ftp) / current_ftp) * 100,
            "features_used": len(selected_feature_names),
            "top_features": selected_feature_names[:5] if selected_feature_names else [],
        }
    
    def predict_performance_readiness(self) -> Dict:
        """Predict readiness for peak performance.
        
        Returns:
            Dictionary with readiness scores and recommendations
        """
        # Get recent training load metrics
        recent_days = 14
        
        # Check if required column exists
        if "start_date_local" not in self.df.columns:
            return {"error": "No start_date_local column found"}
        
        # Convert dates if needed
        if self.df["start_date_local"].dtype == pl.Utf8:
            df = self.df.with_columns(
                pl.col("start_date_local").str.to_datetime()
            )
        else:
            df = self.df
        
        end_date = df["start_date_local"].max()
        if end_date is None:
            return {"error": "No valid dates found"}
        start_date = end_date - timedelta(days=recent_days)
        
        recent_df = df.filter(pl.col("start_date_local") >= start_date)
        
        if recent_df.is_empty():
            return {"error": "No recent activities found"}
        
        # Calculate readiness factors
        readiness_factors = {}
        
        # 1. Training load balance (CTL vs ATL)
        if "icu_fitness" in recent_df.columns and "icu_fatigue" in recent_df.columns:
            avg_fitness = recent_df["icu_fitness"].mean()
            avg_fatigue = recent_df["icu_fatigue"].mean()
            if avg_fitness and avg_fatigue:
                tsb = avg_fitness - avg_fatigue
                # Optimal TSB for performance is typically -10 to +5
                if -10 <= tsb <= 5:
                    readiness_factors["form"] = min(100, 90 + tsb)
                elif tsb > 5:
                    readiness_factors["form"] = max(60, 90 - (tsb - 5) * 2)  # Too fresh
                else:
                    readiness_factors["form"] = max(40, 90 + tsb)  # Too fatigued
            else:
                readiness_factors["form"] = 50
        
        # 2. Training consistency
        expected_activities = recent_days / 3  # Expect activity every 3 days
        actual_activities = len(recent_df)
        consistency_score = min(100, (actual_activities / expected_activities) * 100)
        readiness_factors["consistency"] = consistency_score
        
        # 3. Intensity balance
        if "icu_intensity" in recent_df.columns:
            avg_intensity = recent_df["icu_intensity"].mean()
            if avg_intensity:
                # Optimal intensity around 0.75-0.85
                if 0.75 <= avg_intensity <= 0.85:
                    readiness_factors["intensity"] = 100
                else:
                    diff = min(abs(avg_intensity - 0.75), abs(avg_intensity - 0.85))
                    readiness_factors["intensity"] = max(50, 100 - diff * 200)
            else:
                readiness_factors["intensity"] = 50
        
        # 4. Recovery (based on HRV/resting HR if available)
        if "icu_resting_hr" in recent_df.columns:
            resting_hrs = recent_df["icu_resting_hr"].drop_nulls()
            if len(resting_hrs) > 0:
                # Lower resting HR generally indicates better recovery
                avg_rhr = resting_hrs.mean()
                if avg_rhr < 50:
                    readiness_factors["recovery"] = 100
                elif avg_rhr < 60:
                    readiness_factors["recovery"] = 85
                elif avg_rhr < 70:
                    readiness_factors["recovery"] = 70
                else:
                    readiness_factors["recovery"] = 50
        
        # Calculate overall readiness score
        if readiness_factors:
            overall_readiness = np.mean(list(readiness_factors.values()))
        else:
            overall_readiness = 50
        
        # Generate recommendations
        recommendations = []
        
        if readiness_factors.get("form", 50) < 60:
            if readiness_factors.get("form", 50) < 50:
                recommendations.append("Consider reducing training load - showing signs of fatigue")
            else:
                recommendations.append("Maintain current training load to build form")
        elif readiness_factors.get("form", 50) > 90:
            recommendations.append("Excellent form - ready for peak performance")
        
        if readiness_factors.get("consistency", 0) < 70:
            recommendations.append("Increase training consistency for better adaptations")
        
        if readiness_factors.get("intensity", 50) < 70:
            recommendations.append("Balance training intensity - mix easy and hard efforts")
        
        return {
            "overall_readiness": overall_readiness,
            "readiness_factors": readiness_factors,
            "recommendations": recommendations,
            "assessment_period_days": recent_days,
            "activities_analyzed": len(recent_df),
        }
    
    def predict_race_performance(self, target_date: datetime, race_duration_hours: float = 3.0) -> Dict:
        """Predict race day performance.
        
        Args:
            target_date: Target race date
            race_duration_hours: Expected race duration
            
        Returns:
            Dictionary with race performance predictions
        """
        # Convert dates if needed
        if self.df["start_date_local"].dtype == pl.Utf8:
            df = self.df.with_columns(
                pl.col("start_date_local").str.to_datetime()
            )
        else:
            df = self.df
        
        # Get historical data
        historical_df = df.filter(pl.col("start_date_local") < target_date)
        
        if historical_df.is_empty():
            return {"error": "No historical data available"}
        
        # Calculate days until race
        last_activity = historical_df["start_date_local"].max()
        days_to_race = (target_date - last_activity).days
        
        # Get current fitness metrics
        latest_metrics = historical_df.sort("start_date_local").tail(1)
        
        predictions = {
            "race_date": target_date,
            "days_to_race": days_to_race,
            "race_duration_hours": race_duration_hours,
        }
        
        # Predict power output
        # Get current FTP from unified column
        if "ftp_value" in latest_metrics.columns:
            current_ftp = latest_metrics["ftp_value"][0]
        elif "icu_ftp" in latest_metrics.columns:
            current_ftp = latest_metrics["icu_ftp"][0]
        elif "threshold_power" in latest_metrics.columns:
            current_ftp = latest_metrics["threshold_power"][0]
            if current_ftp:
                # Estimate sustainable power based on duration
                if race_duration_hours <= 1:
                    power_factor = 0.95  # ~95% of FTP for 1 hour
                elif race_duration_hours <= 2:
                    power_factor = 0.90  # ~90% of FTP for 2 hours
                elif race_duration_hours <= 3:
                    power_factor = 0.85  # ~85% of FTP for 3 hours
                elif race_duration_hours <= 4:
                    power_factor = 0.80  # ~80% of FTP for 4 hours
                else:
                    power_factor = 0.75  # ~75% of FTP for longer
                
                predicted_power = current_ftp * power_factor
                predictions["predicted_avg_power"] = predicted_power
                predictions["current_ftp"] = current_ftp
                predictions["power_factor"] = power_factor
        
        # Predict normalized power
        if "predicted_avg_power" in predictions:
            # NP is typically 5-10% higher than average power in races
            predictions["predicted_normalized_power"] = predictions["predicted_avg_power"] * 1.07
        
        # Estimate training load
        if "icu_intensity" in latest_metrics.columns:
            avg_intensity = latest_metrics["icu_intensity"][0]
            if avg_intensity and "predicted_normalized_power" in predictions:
                # Estimate TSS: (duration_sec * NP * IF) / (FTP * 3600) * 100
                duration_sec = race_duration_hours * 3600
                if_value = predictions["predicted_normalized_power"] / predictions.get("current_ftp", 250)
                tss = (duration_sec * predictions["predicted_normalized_power"] * if_value) / (predictions.get("current_ftp", 250) * 3600) * 100
                predictions["estimated_tss"] = tss
        
        # Add confidence based on training
        if "moving_time" in historical_df.columns:
            recent_similar = historical_df.filter(
                (pl.col("moving_time") >= race_duration_hours * 3600 * 0.7) &
                (pl.col("moving_time") <= race_duration_hours * 3600 * 1.3)
            )
        else:
            recent_similar = pl.DataFrame()  # Empty if no moving_time column
        
        if len(recent_similar) >= 3:
            predictions["confidence"] = "High"
            predictions["similar_efforts_count"] = len(recent_similar)
        elif len(recent_similar) >= 1:
            predictions["confidence"] = "Medium"
            predictions["similar_efforts_count"] = len(recent_similar)
        else:
            predictions["confidence"] = "Low"
            predictions["similar_efforts_count"] = 0
            predictions["recommendation"] = "Consider adding more race-duration efforts to training"
        
        return predictions
    
    def forecast_season_trajectory(self, months_ahead: int = 3) -> Dict:
        """Forecast season-long performance trajectory.
        
        Args:
            months_ahead: Number of months to forecast
            
        Returns:
            Dictionary with seasonal forecasts
        """
        # Check if required column exists
        if "start_date_local" not in self.df.columns:
            return {"error": "No start_date_local column found"}
        
        # Convert dates if needed
        if self.df["start_date_local"].dtype == pl.Utf8:
            df = self.df.with_columns(
                pl.col("start_date_local").str.to_datetime()
            )
        else:
            df = self.df
        
        # Calculate monthly aggregates with available columns
        agg_exprs = [pl.len().alias("activity_count")]
        
        if "icu_training_load" in df.columns:
            agg_exprs.append(pl.col("icu_training_load").sum().alias("total_load"))
        # Add FTP data from any available source
        if "icu_ftp" in df.columns:
            agg_exprs.append(pl.col("icu_ftp").max().alias("max_ftp"))
        elif "threshold_power" in df.columns:
            agg_exprs.append(pl.col("threshold_power").max().alias("max_ftp"))
        elif "functional_threshold_power" in df.columns:
            agg_exprs.append(pl.col("functional_threshold_power").max().alias("max_ftp"))
        if "icu_average_watts" in df.columns:
            agg_exprs.append(pl.col("icu_average_watts").mean().alias("avg_power"))
        
        monthly_stats = df.with_columns(
            pl.col("start_date_local").dt.truncate("1mo").alias("month")
        ).group_by("month").agg(agg_exprs).sort("month")
        
        if len(monthly_stats) < 3:
            return {"error": "Insufficient historical data for seasonal forecast"}
        
        # Prepare time series data
        X = np.arange(len(monthly_stats)).reshape(-1, 1)
        
        forecasts = {}
        
        # Forecast different metrics (only those that exist)
        metrics_to_forecast = ["total_load", "activity_count", "max_ftp", "avg_power"]
        
        for metric in metrics_to_forecast:
            if metric in monthly_stats.columns:
                y = monthly_stats[metric].to_numpy()
                
                # Remove nulls
                valid_mask = ~np.isnan(y)
                if valid_mask.sum() < 3:
                    continue
                
                X_valid = X[valid_mask]
                y_valid = y[valid_mask]
                
                # Fit trend model
                model = LinearRegression()
                model.fit(X_valid, y_valid)
                
                # Make predictions
                future_X = np.arange(len(monthly_stats), len(monthly_stats) + months_ahead).reshape(-1, 1)
                future_predictions = model.predict(future_X)
                
                # Calculate trend
                trend_coefficient = model.coef_[0]
                if trend_coefficient > 0:
                    trend = "increasing"
                elif trend_coefficient < 0:
                    trend = "decreasing"
                else:
                    trend = "stable"
                
                forecasts[metric] = {
                    "predictions": future_predictions.tolist(),
                    "trend": trend,
                    "monthly_change": trend_coefficient,
                    "r2_score": model.score(X_valid, y_valid),
                }
        
        # Generate seasonal recommendations
        recommendations = []
        
        if "total_load" in forecasts:
            if forecasts["total_load"]["trend"] == "increasing":
                recommendations.append("Progressive overload detected - ensure adequate recovery")
            elif forecasts["total_load"]["trend"] == "decreasing":
                recommendations.append("Consider gradually increasing training load")
        
        if "activity_count" in forecasts:
            if forecasts["activity_count"]["monthly_change"] < 0:
                recommendations.append("Activity frequency declining - maintain consistency")
        
        return {
            "forecasts": forecasts,
            "months_forecasted": months_ahead,
            "recommendations": recommendations,
            "historical_months": len(monthly_stats),
        }