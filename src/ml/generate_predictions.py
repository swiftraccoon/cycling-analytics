"""Generate performance predictions using trained ML models."""

import sys
import logging
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.storage.database.manager import DatabaseManager
from src.ml.predictions import PerformancePredictor
from src.config import REPORTS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_prediction_plots(predictions):
    """Create interactive plots for predictions."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "FTP Progression Forecast",
            "Performance Readiness",
            "Training Load Trajectory",
            "Confidence Intervals"
        )
    )
    
    # FTP Progression
    if "ftp_progression" in predictions and "predictions" in predictions["ftp_progression"]:
        ftp_data = predictions["ftp_progression"]["predictions"]
        fig.add_trace(
            go.Scatter(
                x=ftp_data["dates"],
                y=ftp_data["ftp_values"],
                mode='lines',
                name='Predicted FTP',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add confidence intervals
        fig.add_trace(
            go.Scatter(
                x=ftp_data["dates"] + ftp_data["dates"][::-1],
                y=ftp_data["confidence_upper"] + ftp_data["confidence_lower"][::-1],
                fill='toself',
                fillcolor='rgba(0,100,255,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Performance Readiness
    if "readiness" in predictions and "readiness_factors" in predictions["readiness"]:
        factors = predictions["readiness"]["readiness_factors"]
        fig.add_trace(
            go.Bar(
                x=list(factors.keys()),
                y=list(factors.values()),
                marker_color=['green' if v > 70 else 'orange' if v > 50 else 'red' 
                             for v in factors.values()],
                name='Readiness Factors'
            ),
            row=1, col=2
        )
    
    # Season Trajectory
    if "season_forecast" in predictions and "forecasts" in predictions["season_forecast"]:
        forecasts = predictions["season_forecast"]["forecasts"]
        if "total_load" in forecasts:
            load_data = forecasts["total_load"]["predictions"]
            months = [f"Month {i+1}" for i in range(len(load_data))]
            fig.add_trace(
                go.Scatter(
                    x=months,
                    y=load_data,
                    mode='lines+markers',
                    name='Training Load Forecast',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )
    
    # Model Confidence
    if "ftp_progression" in predictions and "model_scores" in predictions["ftp_progression"]:
        scores = predictions["ftp_progression"]["model_scores"]
        fig.add_trace(
            go.Bar(
                x=list(scores.keys()),
                y=list(scores.values()),
                marker_color='lightblue',
                name='Model R² Scores'
            ),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=True, title_text="Performance Predictions Dashboard")
    return fig


def main():
    """Generate and display performance predictions."""
    parser = argparse.ArgumentParser(description="Generate cycling performance predictions")
    parser.add_argument("--days", type=int, default=30, help="Days to predict ahead (default: 30)")
    parser.add_argument("--race-date", type=str, help="Target race date (YYYY-MM-DD)")
    parser.add_argument("--race-duration", type=float, help="Expected race duration in hours")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to HTML file")
    args = parser.parse_args()
    
    logger.info("Generating performance predictions...")
    
    # Load data from database (uses default from config)
    db = DatabaseManager()
    activities_df = db.get_activities()
    
    if activities_df.is_empty():
        logger.error("No activities found in database. Please run 'cycling ingest' or 'cycling sync' first.")
        return
    
    logger.info(f"Loaded {len(activities_df)} activities from database")
    
    # Initialize predictor
    predictor = PerformancePredictor(activities_df)
    
    predictions = {}
    
    # 1. FTP Progression Prediction
    logger.info(f"\n{'='*50}")
    logger.info("FTP PROGRESSION FORECAST")
    logger.info("="*50)
    
    ftp_result = predictor.predict_ftp_progression(days_ahead=args.days)
    predictions["ftp_progression"] = ftp_result
    
    if "error" not in ftp_result:
        logger.info(f"Current FTP: {ftp_result['current_ftp']:.0f}W")
        logger.info(f"Predicted FTP in {args.days} days: {ftp_result['predictions']['ftp_values'][-1]:.0f}W")
        logger.info(f"Expected gain: {ftp_result['expected_gain']:.1f}W ({ftp_result['expected_gain_percentage']:.1f}%)")
        logger.info(f"Best model: {ftp_result['best_model']}")
        
        # Show weekly milestones
        logger.info("\nWeekly milestones:")
        for i in [7, 14, 21, 28]:
            if i < len(ftp_result['predictions']['ftp_values']):
                predicted_ftp = ftp_result['predictions']['ftp_values'][i-1]
                gain = predicted_ftp - ftp_result['current_ftp']
                logger.info(f"  Week {i//7}: {predicted_ftp:.0f}W (+{gain:.1f}W)")
    else:
        logger.warning(f"FTP prediction unavailable: {ftp_result['error']}")
    
    # 2. Performance Readiness
    logger.info(f"\n{'='*50}")
    logger.info("PERFORMANCE READINESS ASSESSMENT")
    logger.info("="*50)
    
    readiness = predictor.predict_performance_readiness()
    predictions["readiness"] = readiness
    
    if "error" not in readiness:
        overall = readiness['overall_readiness']
        
        # Determine readiness level
        if overall >= 80:
            status = "PEAK - Ready for maximum efforts"
            color = "🟢"
        elif overall >= 60:
            status = "GOOD - Maintain current training"
            color = "🟡"
        else:
            status = "BUILDING - Focus on recovery"
            color = "🔴"
        
        logger.info(f"Overall Readiness: {overall:.0f}% - {status}")
        
        if "readiness_factors" in readiness:
            logger.info("\nReadiness Factors:")
            for factor, score in readiness['readiness_factors'].items():
                bar_length = int(score / 5)  # Scale to 20 chars
                bar = "█" * bar_length + "░" * (20 - bar_length)
                logger.info(f"  {factor.capitalize():12} [{bar}] {score:.0f}%")
        
        if readiness.get('recommendations'):
            logger.info("\nRecommendations:")
            for i, rec in enumerate(readiness['recommendations'], 1):
                logger.info(f"  {i}. {rec}")
    else:
        logger.warning(f"Readiness assessment unavailable: {readiness.get('error', 'Unknown error')}")
    
    # 3. Race Performance Prediction (if requested)
    if args.race_date:
        logger.info(f"\n{'='*50}")
        logger.info("RACE PERFORMANCE PREDICTION")
        logger.info("="*50)
        
        try:
            race_date = datetime.strptime(args.race_date, "%Y-%m-%d")
            race_duration = args.race_duration or 3.0
            
            race_pred = predictor.predict_race_performance(
                target_date=race_date,
                race_duration_hours=race_duration
            )
            predictions["race"] = race_pred
            
            if "error" not in race_pred:
                logger.info(f"Race Date: {race_date.strftime('%Y-%m-%d')}")
                logger.info(f"Days to Race: {race_pred['days_to_race']}")
                logger.info(f"Race Duration: {race_duration:.1f} hours")
                
                if "predicted_avg_power" in race_pred:
                    logger.info(f"Predicted Average Power: {race_pred['predicted_avg_power']:.0f}W")
                    logger.info(f"Predicted Normalized Power: {race_pred.get('predicted_normalized_power', 0):.0f}W")
                    logger.info(f"Current FTP: {race_pred.get('current_ftp', 0):.0f}W")
                    logger.info(f"Power Factor: {race_pred.get('power_factor', 0):.2f}")
                
                if "estimated_tss" in race_pred:
                    logger.info(f"Estimated TSS: {race_pred['estimated_tss']:.0f}")
                
                logger.info(f"Confidence: {race_pred.get('confidence', 'Unknown')}")
                
                if "recommendation" in race_pred:
                    logger.info(f"Recommendation: {race_pred['recommendation']}")
            else:
                logger.warning(f"Race prediction unavailable: {race_pred['error']}")
        except ValueError:
            logger.error(f"Invalid race date format. Use YYYY-MM-DD")
    
    # 4. Season Trajectory
    logger.info(f"\n{'='*50}")
    logger.info("SEASON TRAJECTORY FORECAST")
    logger.info("="*50)
    
    season_forecast = predictor.forecast_season_trajectory(months_ahead=3)
    predictions["season_forecast"] = season_forecast
    
    if "error" not in season_forecast:
        logger.info(f"Historical months analyzed: {season_forecast['historical_months']}")
        
        if "forecasts" in season_forecast:
            for metric, forecast in season_forecast['forecasts'].items():
                logger.info(f"\n{metric.replace('_', ' ').title()}:")
                logger.info(f"  Trend: {forecast['trend'].upper()}")
                logger.info(f"  Monthly change: {forecast['monthly_change']:.2f}")
                logger.info(f"  Model confidence (R²): {forecast['r2_score']:.3f}")
                
                if forecast["predictions"]:
                    logger.info("  3-month forecast:")
                    for i, pred in enumerate(forecast["predictions"], 1):
                        logger.info(f"    Month {i}: {pred:.1f}")
        
        if season_forecast.get('recommendations'):
            logger.info("\nSeason Recommendations:")
            for rec in season_forecast['recommendations']:
                logger.info(f"  - {rec}")
    else:
        logger.warning(f"Season forecast unavailable: {season_forecast.get('error', 'Unknown error')}")
    
    # 5. Training Insights
    logger.info(f"\n{'='*50}")
    logger.info("TRAINING INSIGHTS")
    logger.info("="*50)
    
    # Recent training summary
    recent_days = 30
    recent_date = datetime.now() - timedelta(days=recent_days)
    recent_activities = activities_df.filter(pl.col("start_date_local") >= recent_date)
    
    if not recent_activities.is_empty():
        logger.info(f"Last {recent_days} days:")
        logger.info(f"  Activities: {len(recent_activities)}")
        
        if "distance" in recent_activities.columns:
            total_distance = recent_activities["distance"].sum()
            if total_distance:
                logger.info(f"  Total distance: {total_distance/1000:.0f} km")
        
        if "moving_time" in recent_activities.columns:
            total_time = recent_activities["moving_time"].sum()
            if total_time:
                logger.info(f"  Total time: {total_time/3600:.1f} hours")
        
        if "icu_training_load" in recent_activities.columns:
            avg_load = recent_activities["icu_training_load"].mean()
            if avg_load:
                logger.info(f"  Average training load: {avg_load:.1f}")
    
    # Save plots if requested
    if args.save_plots:
        try:
            fig = create_prediction_plots(predictions)
            output_path = REPORTS_DIR / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            fig.write_html(str(output_path))
            logger.info(f"\nPlots saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save plots: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info("Prediction generation complete!")
    logger.info("Run 'cycling dashboard' to see visualizations")


if __name__ == "__main__":
    main()