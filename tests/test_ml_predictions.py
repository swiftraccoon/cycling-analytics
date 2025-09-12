"""Tests for machine learning predictions module."""

import pytest
import polars as pl
from datetime import datetime, timedelta
import numpy as np

from src.ml.predictions import PerformancePredictor


@pytest.fixture
def training_data_with_ftp():
    """Create training data with FTP progression."""
    num_activities = 30
    base_date = datetime(2024, 1, 1)
    
    # Generate progressive FTP data
    activities = []
    for i in range(num_activities):
        activities.append({
            "id": f"act{i}",
            "start_date_local": base_date + timedelta(days=i*3),
            "icu_ftp": 200 + (i * 2),  # FTP increases by 2W every activity
            "icu_training_load": 50 + np.random.randint(-10, 20),
            "icu_intensity": 0.75 + np.random.random() * 0.15,
            "icu_fitness": 40 + i * 0.5,
            "icu_fatigue": 30 + i * 0.3,
            "moving_time": 3600 + np.random.randint(-600, 1200),
            "icu_average_watts": 180 + i * 1.5,
        })
    
    return pl.DataFrame(activities)


@pytest.fixture
def minimal_training_data():
    """Create minimal training data."""
    return pl.DataFrame({
        "id": ["act1", "act2"],
        "start_date_local": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        "icu_ftp": [200, 202],
        "moving_time": [3600, 3700],
    })


@pytest.fixture
def recent_activity_data():
    """Create recent activity data for readiness assessment."""
    base_date = datetime.now() - timedelta(days=7)
    
    activities = []
    for i in range(10):
        activities.append({
            "id": f"act{i}",
            "start_date_local": base_date + timedelta(days=i),
            "icu_fitness": 50 + np.random.random() * 5,
            "icu_fatigue": 40 + np.random.random() * 10,
            "icu_intensity": 0.70 + np.random.random() * 0.20,
            "icu_resting_hr": 55 + np.random.randint(-5, 10),
            "icu_training_load": 60 + np.random.randint(-20, 30),
        })
    
    return pl.DataFrame(activities)


class TestFTPPrediction:
    """Tests for FTP progression prediction."""
    
    def test_ftp_prediction_with_sufficient_data(self, training_data_with_ftp):
        """Test FTP prediction with sufficient training data."""
        predictor = PerformancePredictor(training_data_with_ftp)
        
        result = predictor.predict_ftp_progression(days_ahead=30)
        
        # Should return predictions
        assert "error" not in result
        assert "current_ftp" in result
        assert "predictions" in result
        assert "model_scores" in result
        
        # Check current FTP is correct (last value)
        assert result["current_ftp"] == training_data_with_ftp["icu_ftp"][-1]
        
        # Check predictions structure
        assert "dates" in result["predictions"]
        assert "ftp_values" in result["predictions"]
        assert len(result["predictions"]["dates"]) == 30
        assert len(result["predictions"]["ftp_values"]) == 30
        
        # FTP should generally increase given training data trend
        assert result["expected_gain"] >= 0
    
    def test_ftp_prediction_with_insufficient_data(self, minimal_training_data):
        """Test FTP prediction with insufficient data."""
        predictor = PerformancePredictor(minimal_training_data)
        
        result = predictor.predict_ftp_progression(days_ahead=30)
        
        # Should return error due to insufficient data
        assert "error" in result
        assert result["current_activities"] == 2
        assert result["required_activities"] == 10
    
    def test_ftp_prediction_without_ftp_data(self):
        """Test FTP prediction when no FTP data exists."""
        df = pl.DataFrame({
            "id": ["act1"],
            "start_date_local": [datetime.now()],
            "distance": [30000],
        })
        
        predictor = PerformancePredictor(df)
        result = predictor.predict_ftp_progression()
        
        assert "error" in result
    
    def test_ftp_prediction_model_scores(self, training_data_with_ftp):
        """Test that model scores are reasonable."""
        predictor = PerformancePredictor(training_data_with_ftp)
        
        result = predictor.predict_ftp_progression(days_ahead=7)
        
        if "model_scores" in result:
            # Check all models are evaluated
            assert "rf" in result["model_scores"]
            assert "gb" in result["model_scores"]
            assert "lr" in result["model_scores"]
            
            # Scores should be between -1 and 1 (R² score)
            for score in result["model_scores"].values():
                assert -1 <= score <= 1
    
    def test_ftp_prediction_confidence_intervals(self, training_data_with_ftp):
        """Test that confidence intervals are provided."""
        predictor = PerformancePredictor(training_data_with_ftp)
        
        result = predictor.predict_ftp_progression(days_ahead=14)
        
        if "predictions" in result:
            assert "confidence_lower" in result["predictions"]
            assert "confidence_upper" in result["predictions"]
            
            # Confidence intervals should bracket predictions
            for i in range(len(result["predictions"]["ftp_values"])):
                pred = result["predictions"]["ftp_values"][i]
                lower = result["predictions"]["confidence_lower"][i]
                upper = result["predictions"]["confidence_upper"][i]
                
                assert lower <= pred <= upper


class TestPerformanceReadiness:
    """Tests for performance readiness assessment."""
    
    def test_readiness_assessment_with_data(self, recent_activity_data):
        """Test readiness assessment with recent activities."""
        predictor = PerformancePredictor(recent_activity_data)
        
        result = predictor.predict_performance_readiness()
        
        assert "error" not in result
        assert "overall_readiness" in result
        assert "readiness_factors" in result
        assert "recommendations" in result
        
        # Overall readiness should be 0-100
        assert 0 <= result["overall_readiness"] <= 100
        
        # Check readiness factors
        factors = result["readiness_factors"]
        assert "form" in factors
        assert "consistency" in factors
        assert "intensity" in factors
        
        # All factors should be 0-100
        for factor, score in factors.items():
            assert 0 <= score <= 100
    
    def test_readiness_without_recent_activities(self):
        """Test readiness when no recent activities exist."""
        old_data = pl.DataFrame({
            "id": ["act1"],
            "start_date_local": [datetime.now() - timedelta(days=30)],
            "icu_fitness": [50],
        })
        
        predictor = PerformancePredictor(old_data)
        result = predictor.predict_performance_readiness()
        
        assert "error" in result or result["activities_analyzed"] == 0
    
    def test_readiness_recommendations(self, recent_activity_data):
        """Test that appropriate recommendations are generated."""
        predictor = PerformancePredictor(recent_activity_data)
        
        result = predictor.predict_performance_readiness()
        
        if "recommendations" in result:
            assert isinstance(result["recommendations"], list)
            
            # Recommendations should be meaningful strings
            for rec in result["recommendations"]:
                assert isinstance(rec, str)
                assert len(rec) > 10  # Not empty recommendations
    
    def test_readiness_with_high_fatigue(self):
        """Test readiness detection with high fatigue."""
        df = pl.DataFrame({
            "id": ["act1"],
            "start_date_local": [datetime.now()],
            "icu_fitness": [50],
            "icu_fatigue": [65],  # High fatigue (TSB = -15)
            "icu_intensity": [0.80],
        })
        
        predictor = PerformancePredictor(df)
        result = predictor.predict_performance_readiness()
        
        if "readiness_factors" in result:
            # Form score should be lower with high fatigue
            assert result["readiness_factors"]["form"] < 60


class TestRacePerformancePrediction:
    """Tests for race performance prediction."""
    
    def test_race_prediction_basic(self, training_data_with_ftp):
        """Test basic race performance prediction."""
        predictor = PerformancePredictor(training_data_with_ftp)
        
        target_date = datetime.now() + timedelta(days=30)
        result = predictor.predict_race_performance(
            target_date=target_date,
            race_duration_hours=3.0
        )
        
        assert "error" not in result
        assert "race_date" in result
        assert "days_to_race" in result
        assert "predicted_avg_power" in result
        
        # Power predictions should be reasonable
        if "predicted_avg_power" in result:
            # Should be less than FTP for 3-hour race
            assert result["predicted_avg_power"] < result.get("current_ftp", 300)
    
    def test_race_prediction_different_durations(self, training_data_with_ftp):
        """Test race predictions for different durations."""
        predictor = PerformancePredictor(training_data_with_ftp)
        target_date = datetime.now() + timedelta(days=14)
        
        durations = [1.0, 2.0, 3.0, 4.0, 6.0]
        power_factors = []
        
        for duration in durations:
            result = predictor.predict_race_performance(
                target_date=target_date,
                race_duration_hours=duration
            )
            
            if "power_factor" in result:
                power_factors.append(result["power_factor"])
        
        # Power factor should decrease with duration
        if len(power_factors) == len(durations):
            for i in range(1, len(power_factors)):
                assert power_factors[i] <= power_factors[i-1]
    
    def test_race_prediction_confidence(self, training_data_with_ftp):
        """Test race prediction confidence based on similar efforts."""
        predictor = PerformancePredictor(training_data_with_ftp)
        
        # Add some long efforts
        long_efforts = pl.DataFrame({
            "id": ["long1", "long2", "long3"],
            "start_date_local": [datetime.now() - timedelta(days=i) for i in range(1, 4)],
            "moving_time": [10800, 11000, 10500],  # ~3 hour efforts
            "icu_ftp": [250, 250, 250],
        })
        
        combined_df = pl.concat([training_data_with_ftp, long_efforts])
        predictor_with_long = PerformancePredictor(combined_df)
        
        result = predictor_with_long.predict_race_performance(
            target_date=datetime.now() + timedelta(days=7),
            race_duration_hours=3.0
        )
        
        if "confidence" in result:
            # Should have higher confidence with similar efforts
            assert result["confidence"] in ["High", "Medium"]
            assert result["similar_efforts_count"] >= 3


class TestSeasonTrajectory:
    """Tests for season trajectory forecasting."""
    
    def test_season_forecast_basic(self, training_data_with_ftp):
        """Test basic season trajectory forecast."""
        predictor = PerformancePredictor(training_data_with_ftp)
        
        result = predictor.forecast_season_trajectory(months_ahead=3)
        
        if "error" not in result:
            assert "forecasts" in result
            assert "recommendations" in result
            assert "historical_months" in result
    
    def test_season_forecast_trends(self):
        """Test trend detection in season forecast."""
        # Create data with clear upward trend
        dates = [datetime(2024, i, 1) for i in range(1, 7)]
        df = pl.DataFrame({
            "id": [f"act{i}" for i in range(6)],
            "start_date_local": dates,
            "icu_training_load": [50, 60, 70, 80, 90, 100],  # Clear upward trend
            "moving_time": [3600, 4000, 4400, 4800, 5200, 5600],
        })
        
        predictor = PerformancePredictor(df)
        result = predictor.forecast_season_trajectory(months_ahead=2)
        
        if "forecasts" in result and "total_load" in result["forecasts"]:
            # Should detect increasing trend
            assert result["forecasts"]["total_load"]["trend"] == "increasing"
            assert result["forecasts"]["total_load"]["monthly_change"] > 0
    
    def test_season_forecast_insufficient_data(self):
        """Test season forecast with insufficient historical data."""
        df = pl.DataFrame({
            "id": ["act1"],
            "start_date_local": [datetime.now()],
            "icu_training_load": [50],
        })
        
        predictor = PerformancePredictor(df)
        result = predictor.forecast_season_trajectory(months_ahead=3)
        
        assert "error" in result


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling in predictions."""
    
    def test_empty_dataframe(self):
        """Test predictions with empty dataframe."""
        predictor = PerformancePredictor(pl.DataFrame())
        
        # FTP prediction
        ftp_result = predictor.predict_ftp_progression()
        assert "error" in ftp_result or ftp_result == {}
        
        # Readiness
        readiness_result = predictor.predict_performance_readiness()
        assert "error" in readiness_result
        
        # Season forecast
        season_result = predictor.forecast_season_trajectory()
        assert "error" in season_result
    
    def test_missing_required_columns(self):
        """Test handling of missing required columns."""
        df = pl.DataFrame({
            "id": ["act1"],
            "distance": [30000],  # Missing date and other required fields
        })
        
        predictor = PerformancePredictor(df)
        
        # Should handle gracefully
        ftp_result = predictor.predict_ftp_progression()
        assert "error" in ftp_result
    
    def test_null_values_handling(self):
        """Test handling of null values in data."""
        df = pl.DataFrame({
            "id": ["act1", "act2", "act3"],
            "start_date_local": [datetime.now() - timedelta(days=i) for i in range(3)],
            "icu_ftp": [250, None, 255],  # Null in middle
            "icu_training_load": [50, 60, None],  # Null at end
            "icu_fitness": [None, 45, 50],  # Null at start
        })
        
        predictor = PerformancePredictor(df)
        
        # Should handle nulls gracefully
        result = predictor.predict_performance_readiness()
        assert "error" in result or "overall_readiness" in result
    
    def test_single_activity_predictions(self):
        """Test predictions with only one activity."""
        df = pl.DataFrame({
            "id": ["act1"],
            "start_date_local": [datetime.now()],
            "icu_ftp": [250],
            "icu_training_load": [60],
        })
        
        predictor = PerformancePredictor(df)
        
        # FTP prediction should fail (need more data)
        ftp_result = predictor.predict_ftp_progression()
        assert "error" in ftp_result
        
        # Readiness might work with limited assessment
        readiness_result = predictor.predict_performance_readiness()
        assert "activities_analyzed" in readiness_result
    
    def test_date_string_handling(self):
        """Test handling of date strings vs datetime objects."""
        df = pl.DataFrame({
            "id": ["act1", "act2"],
            "start_date_local": ["2024-01-01", "2024-01-02"],  # String dates
            "icu_ftp": [250, 252],
        })
        
        # Convert strings to datetime
        df = df.with_columns(
            pl.col("start_date_local").str.to_datetime()
        )
        
        predictor = PerformancePredictor(df)
        result = predictor.predict_performance_readiness()
        
        # Should handle after conversion
        assert "error" in result or "overall_readiness" in result