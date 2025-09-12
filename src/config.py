"""Centralized configuration for the cycling analytics platform."""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Database configuration
DATABASE_PATH = os.environ.get(
    "CYCLING_DB_PATH",
    str(DATA_DIR / "silver" / "cycling_analytics.db")
)

# Data layer paths
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

# Model paths
MODELS_DIR = DATA_DIR / "models"
MODEL_PICKLE_PATH = MODELS_DIR / "performance_predictor.pkl"

# Report paths
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure directories exist
for directory in [BRONZE_DIR, SILVER_DIR, GOLD_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API configuration
INTERVALS_ICU_BASE_URL = "https://intervals.icu"
INTERVALS_ICU_STATE_FILE = DATA_DIR / "intervals_icu_state.json"

# Intervals.icu settings
INTERVALS_ICU_ATHLETE_ID = os.environ.get("INTERVALS_ICU_ATHLETE_ID")
INTERVALS_ICU_API_KEY = os.environ.get("INTERVALS_ICU_API_KEY")

# Garmin Connect settings
GARMIN_EMAIL = os.environ.get("GARMIN_EMAIL")
GARMIN_PASSWORD = os.environ.get("GARMIN_PASSWORD")
GARMIN_TOKENSTORE = os.environ.get("GARMIN_TOKENSTORE", str(Path.home() / ".garminconnect"))
GARMIN_STATE_FILE = DATA_DIR / "garmin_sync_state.json"
GARMIN_FIT_DIR = BRONZE_DIR / "fit_files"

# Processing configuration
DEFAULT_BATCH_SIZE = 100
MAX_RETRIES = 3
RATE_LIMIT_CALLS_PER_MINUTE = 90

# Cache configuration
CACHE_TTL_SECONDS = 300  # 5 minutes for dashboard cache