"""Train machine learning models on cycling data."""

import sys
import logging
import pickle
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.storage.database.manager import DatabaseManager
from src.ml.predictions import PerformancePredictor
from src.config import MODELS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Train ML models on cycling data."""
    logger.info("Starting ML model training...")
    
    # Load data from database (uses default from config)
    db = DatabaseManager()
    activities_df = db.get_activities()
    
    if activities_df.is_empty():
        logger.error("No activities found in database. Please run 'cycling ingest' or 'cycling sync' first.")
        return
    
    logger.info(f"Loaded {len(activities_df)} activities from database")
    
    # Initialize predictor
    predictor = PerformancePredictor(activities_df)
    
    # Check if we have enough data for training
    if "icu_ftp" not in activities_df.columns:
        logger.warning("No FTP data available. Models require FTP data for training.")
        logger.info("Sync data from Intervals.icu to get FTP values: 'cycling sync'")
        return
    
    ftp_count = activities_df.filter(pl.col("icu_ftp").is_not_null()).shape[0]
    logger.info(f"Found {ftp_count} activities with FTP data")
    
    if ftp_count < 10:
        logger.warning(f"Insufficient FTP data for training (have {ftp_count}, need at least 10)")
        logger.info("Record more activities with power data or sync from Intervals.icu")
        return
    
    # Prepare training data
    logger.info("Preparing training data...")
    X, y, processed_df = predictor.prepare_training_data()
    
    if X is None or len(X) < 10:
        logger.error("Failed to prepare training data. Check data quality.")
        return
    
    logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
    
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # Don't shuffle time series
    )
    
    # Train FTP progression model
    logger.info("Training FTP progression models...")
    ftp_result = predictor.predict_ftp_progression(days_ahead=30)
    
    if "error" not in ftp_result:
        logger.info("FTP Progression Model Results:")
        logger.info(f"  Current FTP: {ftp_result['current_ftp']:.0f}W")
        logger.info(f"  Expected gain (30 days): {ftp_result['expected_gain']:.1f}W ({ftp_result['expected_gain_percentage']:.1f}%)")
        logger.info(f"  Best model: {ftp_result['best_model']}")
        
        for model_name, score in ftp_result['model_scores'].items():
            logger.info(f"  {model_name} R² score: {score:.3f}")
        
        # Save trained models
        # Save the predictor with trained models
        model_path = MODELS_DIR / "performance_predictor.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'models': predictor.models,
                'scalers': predictor.scalers,
                'training_date': datetime.now(),
                'training_samples': len(X),
                'features_count': X.shape[1],
                'ftp_range': (y.min(), y.max()),
                'model_scores': ftp_result.get('model_scores', {})
            }, f)
        
        logger.info(f"Models saved to {model_path}")
    else:
        logger.error(f"FTP model training failed: {ftp_result['error']}")
    
    # Performance readiness assessment
    logger.info("\nAnalyzing performance readiness...")
    readiness = predictor.predict_performance_readiness()
    
    if "error" not in readiness:
        logger.info("Performance Readiness:")
        logger.info(f"  Overall readiness: {readiness['overall_readiness']:.0f}%")
        
        if "readiness_factors" in readiness:
            logger.info("  Readiness factors:")
            for factor, score in readiness['readiness_factors'].items():
                logger.info(f"    {factor.capitalize()}: {score:.0f}%")
        
        if readiness.get('recommendations'):
            logger.info("  Recommendations:")
            for rec in readiness['recommendations']:
                logger.info(f"    - {rec}")
    
    # Season trajectory forecast
    logger.info("\nForecasting season trajectory...")
    season_forecast = predictor.forecast_season_trajectory(months_ahead=3)
    
    if "error" not in season_forecast:
        logger.info("Season Trajectory Forecast:")
        logger.info(f"  Historical months analyzed: {season_forecast['historical_months']}")
        logger.info(f"  Months forecasted: {season_forecast['months_forecasted']}")
        
        if "forecasts" in season_forecast:
            for metric, forecast in season_forecast['forecasts'].items():
                logger.info(f"  {metric}:")
                logger.info(f"    Trend: {forecast['trend']}")
                logger.info(f"    Monthly change: {forecast['monthly_change']:.2f}")
                logger.info(f"    R² score: {forecast['r2_score']:.3f}")
    
    # Summary statistics
    logger.info("\n" + "="*50)
    logger.info("Training Summary:")
    logger.info(f"  Activities processed: {len(activities_df)}")
    logger.info(f"  Activities with FTP: {ftp_count}")
    logger.info(f"  Training samples: {len(X)}")
    logger.info(f"  Features used: {X.shape[1]}")
    logger.info(f"  Models trained: {len(predictor.models)}")
    
    # Data quality assessment
    missing_features = []
    recommended_features = [
        "icu_ftp", "icu_training_load", "icu_fitness", "icu_fatigue",
        "icu_intensity", "icu_average_watts", "icu_normalized_watts"
    ]
    
    for feature in recommended_features:
        if feature not in activities_df.columns:
            missing_features.append(feature)
        elif activities_df.filter(pl.col(feature).is_not_null()).shape[0] < 10:
            missing_features.append(f"{feature} (insufficient data)")
    
    if missing_features:
        logger.info("\nTo improve model accuracy, consider adding:")
        for feature in missing_features:
            logger.info(f"  - {feature}")
        logger.info("Sync from Intervals.icu to get these metrics: 'cycling sync'")
    
    logger.info("\nTraining complete! Run 'cycling predict' to see predictions.")


if __name__ == "__main__":
    main()