#!/usr/bin/env python3
"""Train machine learning models on cycling data using Garmin power metrics."""

import sys
import logging
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import polars as pl
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.storage.database.manager import DatabaseManager
from src.config import MODELS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def estimate_ftp_from_power_data(df: pl.DataFrame) -> pl.DataFrame:
    """Estimate FTP from normalized power data.
    
    Uses the concept that FTP is approximately the power you can sustain for ~1 hour.
    We'll use a rolling window approach to estimate this.
    """
    if "normalized_power" not in df.columns or "duration" not in df.columns:
        return df
    
    # Filter for activities with power data
    power_df = df.filter(pl.col("normalized_power").is_not_null())
    
    if power_df.is_empty():
        return df
    
    # Sort by date
    power_df = power_df.sort("start_date_local")
    
    # Calculate estimated FTP using different methods
    ftp_estimates = []
    
    for row in power_df.iter_rows(named=True):
        estimates = []
        
        # Method 1: 95% of normalized power for efforts 20-60 minutes
        duration_mins = (row.get("duration") or row.get("moving_time", 0)) / 60
        np_value = row.get("normalized_power")
        
        if np_value and 20 <= duration_mins <= 60:
            estimates.append(np_value * 0.95)
        elif np_value and duration_mins > 60:
            estimates.append(np_value * 1.0)  # Longer efforts are closer to FTP
        elif np_value:
            estimates.append(np_value * 0.90)  # Shorter efforts, more conservative
        
        # Use the estimate or None
        if estimates:
            ftp_estimates.append(np.mean(estimates))
        else:
            ftp_estimates.append(None)
    
    # Add estimated FTP to dataframe
    power_df = power_df.with_columns(
        pl.Series("estimated_ftp", ftp_estimates)
    )
    
    # Apply rolling max to get best recent FTP estimate
    power_df = power_df.with_columns(
        pl.col("estimated_ftp").fill_null(strategy="forward").rolling_max(
            window_size=20,
            min_periods=1
        ).alias("rolling_ftp")
    )
    
    # Join back to original dataframe
    result = df.join(
        power_df.select(["id", "estimated_ftp", "rolling_ftp"]),
        on="id",
        how="left"
    )
    
    return result


def prepare_power_training_data(df: pl.DataFrame):
    """Prepare training data for power progression models using ALL available data."""
    
    # Use actual FTP from FIT files first, then icu_ftp, then estimate
    if "threshold_power" in df.columns and not df["threshold_power"].is_null().all():
        logger.info("Using FTP from FIT files (threshold_power)")
        df = df.with_columns(
            pl.col("threshold_power").alias("target_ftp")
        )
    elif "functional_threshold_power" in df.columns and not df["functional_threshold_power"].is_null().all():
        logger.info("Using FTP from Garmin (functional_threshold_power)")
        df = df.with_columns(
            pl.col("functional_threshold_power").alias("target_ftp")
        )
    elif "icu_ftp" in df.columns and not df["icu_ftp"].is_null().all():
        logger.info("Using FTP from Intervals.icu")
        df = df.with_columns(
            pl.col("icu_ftp").alias("target_ftp")
        )
    else:
        logger.info("No FTP data found, estimating from power metrics...")
        df = estimate_ftp_from_power_data(df)
        if "rolling_ftp" in df.columns:
            df = df.with_columns(
                pl.col("rolling_ftp").alias("target_ftp")
            )
    
    # Filter for activities with power data
    power_df = df.filter(
        (pl.col("normalized_power").is_not_null()) |
        (pl.col("average_power").is_not_null())
    )
    
    if power_df.is_empty():
        return None, None, None
    
    # Sort by date
    power_df = power_df.sort("start_date_local")
    
    # Create features
    features = []
    targets = []
    
    for i, row in enumerate(power_df.iter_rows(named=True)):
        feature_dict = {}
        
        # Basic power metrics
        feature_dict["avg_power"] = row.get("average_power") or 0
        feature_dict["normalized_power"] = row.get("normalized_power") or 0
        feature_dict["max_power"] = row.get("max_power") or 0
        
        # Advanced power dynamics from FIT files
        feature_dict["left_right_balance"] = row.get("left_right_balance") or 50.0
        feature_dict["left_torque_effectiveness"] = row.get("avg_left_torque_effectiveness") or 0
        feature_dict["right_torque_effectiveness"] = row.get("avg_right_torque_effectiveness") or 0
        feature_dict["left_pedal_smoothness"] = row.get("avg_left_pedal_smoothness") or 0
        feature_dict["right_pedal_smoothness"] = row.get("avg_right_pedal_smoothness") or 0
        
        # Cadence metrics (now properly captured from FIT)
        feature_dict["avg_cadence"] = row.get("avg_cadence") or row.get("average_cadence") or 0
        feature_dict["max_cadence"] = row.get("max_cadence") or 0
        
        # Temperature as environmental factor
        feature_dict["avg_temperature"] = row.get("avg_temperature") or 20  # Default to 20°C
        feature_dict["max_temperature"] = row.get("max_temperature") or 25
        
        # Training effect scores
        feature_dict["training_effect"] = row.get("training_effect") or row.get("total_training_effect") or 0
        feature_dict["anaerobic_effect"] = row.get("anaerobic_training_effect") or row.get("total_anaerobic_training_effect") or 0
        
        # Work and energy
        feature_dict["total_work_kj"] = (row.get("total_work") or 0) / 1000  # Convert to kJ
        
        # Heart rate metrics
        feature_dict["avg_hr"] = row.get("average_heartrate") or row.get("avg_heart_rate") or 0
        feature_dict["max_hr"] = row.get("max_heartrate") or row.get("max_heart_rate") or 0
        
        # Calculate HR/Power ratio (efficiency indicator)
        if feature_dict["avg_hr"] > 0 and feature_dict["avg_power"] > 0:
            feature_dict["efficiency_factor"] = feature_dict["avg_power"] / feature_dict["avg_hr"]
        else:
            feature_dict["efficiency_factor"] = 0
        
        # Intensity metrics
        if row.get("normalized_power") and row.get("average_power"):
            feature_dict["variability_index"] = row["normalized_power"] / row["average_power"]
        else:
            feature_dict["variability_index"] = 1.0
        
        # Use actual intensity factor if available
        feature_dict["intensity_factor"] = row.get("intensity_factor") or 0
        
        # Duration and load
        feature_dict["duration_hours"] = (row.get("moving_time") or 0) / 3600
        feature_dict["distance_km"] = (row.get("distance") or 0) / 1000
        
        # Calculate training stress score if possible
        if row.get("normalized_power") and row.get("target_ftp") and row.get("moving_time"):
            intensity_factor = row["normalized_power"] / row["target_ftp"]
            tss = (row["moving_time"] * row["normalized_power"] * intensity_factor) / (row["target_ftp"] * 3600) * 100
            feature_dict["calculated_tss"] = min(tss, 500)  # Cap at 500
        else:
            feature_dict["calculated_tss"] = 0
        
        # Historical features (last 7, 14, 28 days)
        if i > 0:
            lookback_7 = max(0, i - 7)
            lookback_14 = max(0, i - 14)
            lookback_28 = max(0, i - 28)
            
            recent_7 = power_df[lookback_7:i]
            recent_14 = power_df[lookback_14:i]
            recent_28 = power_df[lookback_28:i]
            
            feature_dict["avg_np_7d"] = recent_7["normalized_power"].mean() or 0
            feature_dict["avg_np_14d"] = recent_14["normalized_power"].mean() or 0
            feature_dict["avg_np_28d"] = recent_28["normalized_power"].mean() or 0
            
            feature_dict["total_tss_7d"] = len(recent_7) * 50  # Approximate
            feature_dict["total_tss_14d"] = len(recent_14) * 50
            feature_dict["total_tss_28d"] = len(recent_28) * 50
        else:
            for period in ["7d", "14d", "28d"]:
                feature_dict[f"avg_np_{period}"] = feature_dict["normalized_power"]
                feature_dict[f"total_tss_{period}"] = 0
        
        # Add target (next FTP or current if improving)
        target = row.get("target_ftp") or row.get("estimated_ftp") or row.get("normalized_power")
        
        if target:
            features.append(feature_dict)
            targets.append(target)
    
    if not features:
        return None, None, None
    
    # Convert to arrays
    feature_names = list(features[0].keys())
    X = np.array([[f[k] for k in feature_names] for f in features])
    y = np.array(targets)
    
    return X, y, feature_names


def train_power_models(X, y, feature_names):
    """Train multiple models for power progression."""
    
    # Split data
    tscv = TimeSeriesSplit(n_splits=3)
    
    models = {
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boost": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "linear": LinearRegression()
    }
    
    results = {}
    
    for name, model in models.items():
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val_scaled)
            score = r2_score(y_val, y_pred)
            scores.append(score)
        
        avg_score = np.mean(scores)
        results[name] = {
            "model": model,
            "score": avg_score,
            "scores": scores
        }
        
        logger.info(f"{name}: R² = {avg_score:.3f}")
    
    # Train final model on all data
    best_model_name = max(results, key=lambda k: results[k]["score"])
    best_model = models[best_model_name]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    best_model.fit(X_scaled, y)
    
    # Feature importance for tree-based models
    feature_importance = None
    if hasattr(best_model, "feature_importances_"):
        feature_importance = dict(zip(feature_names, best_model.feature_importances_))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    return {
        "model": best_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "feature_importance": feature_importance,
        "results": results,
        "best_model": best_model_name
    }


def main():
    """Train ML models on cycling data using Garmin power metrics."""
    logger.info("Starting ML model training with Garmin power data...")
    
    # Load data from database
    db = DatabaseManager()
    activities_df = db.get_activities()
    
    if activities_df.is_empty():
        logger.error("No activities found in database.")
        return
    
    logger.info(f"Loaded {len(activities_df)} activities from database")
    
    # Check for power data
    power_count = 0
    if "normalized_power" in activities_df.columns:
        power_count = activities_df.filter(pl.col("normalized_power").is_not_null()).shape[0]
    elif "average_power" in activities_df.columns:
        power_count = activities_df.filter(pl.col("average_power").is_not_null()).shape[0]
    
    logger.info(f"Found {power_count} activities with power data")
    
    if power_count < 10:
        logger.warning(f"Insufficient power data for training (have {power_count}, need at least 10)")
        logger.info("Record more activities with power meter")
        return
    
    # Prepare training data
    logger.info("Preparing training data...")
    X, y, feature_names = prepare_power_training_data(activities_df)
    
    if X is None or len(X) < 10:
        logger.error("Failed to prepare training data.")
        return
    
    logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
    logger.info(f"Target (FTP) range: {y.min():.0f}W - {y.max():.0f}W")
    
    # Train models
    logger.info("Training power progression models...")
    model_results = train_power_models(X, y, feature_names)
    
    # Display results
    logger.info("\n" + "="*50)
    logger.info("TRAINING RESULTS")
    logger.info("="*50)
    
    logger.info(f"Best Model: {model_results['best_model']}")
    logger.info(f"Cross-validation scores:")
    for name, result in model_results["results"].items():
        logger.info(f"  {name}: {result['score']:.3f} (std: {np.std(result['scores']):.3f})")
    
    if model_results["feature_importance"]:
        logger.info("\nTop 10 Most Important Features:")
        for i, (feature, importance) in enumerate(list(model_results["feature_importance"].items())[:10]):
            logger.info(f"  {i+1}. {feature}: {importance:.3f}")
    
    # Save models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "power_progression_model.pkl"
    
    with open(model_path, "wb") as f:
        pickle.dump(model_results, f)
    
    logger.info(f"\nModel saved to: {model_path}")
    
    # Make a sample prediction
    logger.info("\nSample Prediction:")
    last_sample = X[-1:].reshape(1, -1)
    last_sample_scaled = model_results["scaler"].transform(last_sample)
    predicted_ftp = model_results["model"].predict(last_sample_scaled)[0]
    current_ftp = y[-1]
    
    logger.info(f"  Current estimated FTP: {current_ftp:.0f}W")
    logger.info(f"  Predicted next FTP: {predicted_ftp:.0f}W")
    logger.info(f"  Expected change: {predicted_ftp - current_ftp:+.0f}W")
    
    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()