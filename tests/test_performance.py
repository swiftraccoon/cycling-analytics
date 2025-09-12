"""Tests for performance analytics module."""

import pytest
import polars as pl
from datetime import datetime, timedelta
import numpy as np

from src.analytics.performance import PerformanceAnalyzer


@pytest.fixture
def sample_training_data():
    """Create sample training data for testing."""
    base_date = datetime(2024, 1, 1)
    num_activities = 50
    
    # Generate progressive training data
    dates = [base_date + timedelta(days=i*2) for i in range(num_activities)]
    
    return pl.DataFrame({
        "id": [f"act{i}" for i in range(num_activities)],
        "start_date_local": dates,
        "name": [f"Ride {i}" for i in range(num_activities)],
        "type": ["Ride"] * num_activities,
        "moving_time": [3600 + i * 60 for i in range(num_activities)],  # Increasing duration
        "distance": [30000 + i * 500 for i in range(num_activities)],  # Increasing distance
        "total_elevation_gain": [300 + i * 10 for i in range(num_activities)],
        "average_heartrate": [140 + (i % 20) for i in range(num_activities)],
        "icu_average_watts": [200 + i for i in range(num_activities)],
        "icu_normalized_watts": [210 + i for i in range(num_activities)],
        "icu_training_load": [50 + i * 2 for i in range(num_activities)],
        "icu_intensity": [0.75 + (i % 10) * 0.01 for i in range(num_activities)],
        "icu_ftp": [250 + i // 10 * 5 for i in range(num_activities)],  # FTP increases every 10 activities
        "icu_fitness": [40 + i * 0.5 for i in range(num_activities)],
        "icu_fatigue": [30 + i * 0.3 for i in range(num_activities)],
        "z1_secs": [600] * num_activities,
        "z2_secs": [1200] * num_activities,
        "z3_secs": [900] * num_activities,
        "z4_secs": [600] * num_activities,
        "z5_secs": [300] * num_activities,
        "z6_secs": [0] * num_activities,
        "z7_secs": [0] * num_activities,
    })


def test_calculate_ftp_progression(sample_training_data):
    """Test FTP progression calculation."""
    analyzer = PerformanceAnalyzer(sample_training_data)
    
    ftp_data = analyzer.calculate_ftp_progression()
    
    assert not ftp_data.is_empty()
    assert "start_date_local" in ftp_data.columns
    assert "icu_ftp" in ftp_data.columns
    
    # Check that FTP values are unique and sorted
    ftp_values = ftp_data["icu_ftp"].to_list()
    assert ftp_values == sorted(set(ftp_values))
    
    # Check FTP progression
    assert ftp_values[-1] > ftp_values[0]  # FTP should increase


def test_calculate_training_load(sample_training_data):
    """Test training load calculation."""
    analyzer = PerformanceAnalyzer(sample_training_data)
    
    metrics = analyzer.calculate_training_load(days=42)
    
    assert "ctl" in metrics
    assert "atl" in metrics
    assert "tsb" in metrics
    assert "total_load" in metrics
    assert "activities" in metrics
    assert "total_time_hours" in metrics
    
    # TSB should be CTL - ATL
    if metrics["ctl"] and metrics["atl"]:
        expected_tsb = metrics["ctl"] - metrics["atl"]
        assert abs(metrics["tsb"] - expected_tsb) < 0.1


def test_calculate_power_zones(sample_training_data):
    """Test power zone calculation."""
    analyzer = PerformanceAnalyzer(sample_training_data)
    
    # Test with specific FTP
    zones = analyzer.calculate_power_zones(ftp=250)
    
    assert zones["ftp"] == 250
    assert "zones" in zones
    assert "zone_times_hours" in zones
    
    # Check zone boundaries
    assert zones["zones"]["Z1 Recovery"] == (0, 137.5)  # 55% of FTP
    assert zones["zones"]["Z4 Threshold"][0] == 225  # 90% of FTP
    
    # Check zone times
    assert len(zones["zone_times_hours"]) > 0


def test_calculate_efficiency_metrics(sample_training_data):
    """Test efficiency metrics calculation."""
    analyzer = PerformanceAnalyzer(sample_training_data)
    
    metrics = analyzer.calculate_efficiency_metrics()
    
    assert "avg_watts_per_beat" in metrics
    
    # Check watts per beat is reasonable
    if metrics["avg_watts_per_beat"]:
        assert 1 < metrics["avg_watts_per_beat"] < 3  # Reasonable range


def test_get_best_efforts(sample_training_data):
    """Test getting best efforts."""
    analyzer = PerformanceAnalyzer(sample_training_data)
    
    best_power = analyzer.get_best_efforts(metric="icu_normalized_watts", top_n=5)
    
    assert len(best_power) == 5
    assert "icu_normalized_watts" in best_power.columns
    
    # Check that efforts are sorted descending
    power_values = best_power["icu_normalized_watts"].to_list()
    assert power_values == sorted(power_values, reverse=True)


def test_calculate_weekly_summary(sample_training_data):
    """Test weekly summary calculation."""
    analyzer = PerformanceAnalyzer(sample_training_data)
    
    weekly = analyzer.calculate_weekly_summary(weeks=12)
    
    assert not weekly.is_empty()
    assert "week" in weekly.columns
    assert "activities" in weekly.columns
    assert "total_hours" in weekly.columns
    assert "total_km" in weekly.columns
    
    # Check that weeks are sorted
    weeks = weekly["week"].to_list()
    assert weeks == sorted(weeks)


def test_empty_dataframe():
    """Test handling of empty DataFrame."""
    analyzer = PerformanceAnalyzer(pl.DataFrame())
    
    ftp_data = analyzer.calculate_ftp_progression()
    assert ftp_data.is_empty()
    
    metrics = analyzer.calculate_training_load()
    assert metrics["activities"] == 0
    
    zones = analyzer.calculate_power_zones()
    assert zones == {}


def test_missing_columns():
    """Test handling of missing columns."""
    minimal_df = pl.DataFrame({
        "id": ["act1"],
        "start_date_local": [datetime.now()],
        "moving_time": [3600],
        "distance": [30000],
    })
    
    analyzer = PerformanceAnalyzer(minimal_df)
    
    # Should handle missing FTP column
    ftp_data = analyzer.calculate_ftp_progression()
    assert ftp_data.is_empty()
    
    # Should handle missing power columns
    zones = analyzer.calculate_power_zones()
    assert zones == {}
    
    # Should handle missing efficiency columns
    efficiency = analyzer.calculate_efficiency_metrics()
    assert efficiency == {}