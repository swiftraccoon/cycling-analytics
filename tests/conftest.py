"""Pytest configuration and fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
import polars as pl
from datetime import datetime, timedelta
import random


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="cycling_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def bronze_layer(test_data_dir):
    """Set up bronze layer directory structure."""
    bronze_path = test_data_dir / "bronze"
    (bronze_path / "incoming").mkdir(parents=True, exist_ok=True)
    (bronze_path / "archive").mkdir(parents=True, exist_ok=True)
    return bronze_path


@pytest.fixture
def sample_csv_data():
    """Generate sample CSV data matching Intervals.icu format."""
    base_date = datetime(2024, 1, 1)
    num_activities = 20
    
    activities = []
    for i in range(num_activities):
        activity = {
            "id": f"i{1000 + i}",
            "start_date_local": base_date + timedelta(days=i),
            "icu_sync_date": base_date + timedelta(days=i, hours=1),
            "name": f"Morning Ride {i}",
            "type": random.choice(["Ride", "VirtualRide", "Run"]),
            "moving_time": 3600 + random.randint(-600, 1800),
            "distance": 30000 + random.randint(-5000, 10000),
            "total_elevation_gain": 300 + random.randint(-100, 200),
            "average_speed": 7.5 + random.random() * 2,
            "max_speed": 12 + random.random() * 3,
            "average_heartrate": 135 + random.randint(-10, 20),
            "max_heartrate": 165 + random.randint(-10, 15),
            "average_cadence": 85 + random.randint(-5, 10),
            "icu_average_watts": 200 + random.randint(-30, 50),
            "icu_normalized_watts": 210 + random.randint(-30, 50),
            "icu_training_load": 50 + random.randint(-10, 30),
            "icu_intensity": 0.70 + random.random() * 0.2,
            "icu_ftp": 250 + (i // 5) * 5,  # FTP increases every 5 activities
            "calories": 500 + random.randint(-50, 150),
        }
        activities.append(activity)
    
    return pl.DataFrame(activities)


@pytest.fixture
def create_test_csv(bronze_layer, sample_csv_data):
    """Create a test CSV file in the bronze layer."""
    def _create_csv(filename="test_activities.csv", location="incoming"):
        file_path = bronze_layer / location / filename
        sample_csv_data.write_csv(file_path)
        return file_path
    
    return _create_csv