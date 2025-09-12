# Cycling Analytics Platform

Analyzes cycling performance data from Intervals.icu and Garmin Connect with machine learning predictions and interactive visualizations. Supports detailed FIT file analysis for second-by-second data.

## Features

- Direct Intervals.icu API sync with incremental updates
- Garmin Connect sync with FIT file downloads for detailed analysis
- Second-by-second data analysis from FIT files (power, HR, cadence, etc.)
- Performance analytics: FTP progression, training load (CTL/ATL/TSB), power zones
- ML predictions for FTP progression and race performance
- Interactive Streamlit dashboard with charts and heatmaps
- Data export to Excel and CSV
- Automatic deduplication and data integrity preservation

## Installation

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/swiftraccoon/cycling.git
cd cycling
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

## Quick Start

### Option 1: Garmin Connect Sync (Recommended for ML)

1. Configure credentials in `.env` file (see Configuration section)

2. Sync activities with FIT files:

```bash
# Basic sync
cycling garmin

# With detailed FIT analysis for ML
cycling garmin --analyze-fit
```

3. Train ML models and view predictions:

```bash
cycling train
cycling predict
cycling dashboard
```

### Option 2: Intervals.icu API Sync

1. Get API credentials from Intervals.icu Settings → Developer Settings
2. Configure credentials in `.env` file (see Configuration section)
3. Sync activities:

```bash
cycling sync
```

4. Launch dashboard:

```bash
cycling dashboard
```

### Option 3: CSV Import

1. Export activities from Intervals.icu as CSV
2. Place CSV files in `data/bronze/incoming/`
3. Run:

```bash
cycling ingest
cycling dashboard
```

## Project Structure

```
cycling/
├── data/                   # Data storage
│   ├── bronze/            # Raw CSV files
│   │   ├── incoming/      # New CSV files to process
│   │   ├── archive/       # Processed CSV files
│   │   └── fit_files/     # Downloaded FIT files from Garmin
│   ├── silver/            # Processed database
│   └── gold/              # Analytics outputs
├── src/
│   ├── etl/               # Data processing pipeline
│   ├── analytics/         # Performance calculations
│   ├── ml/                # Machine learning models
│   ├── integrations/      # API integrations
│   └── dashboard/         # Streamlit web interface
└── tests/                 # Test suite
```

## Usage

### Sync Latest Activities

#### From Garmin Connect (with FIT files):

```bash
# Incremental sync
cycling garmin

# Full sync (last 90 days)
cycling garmin --force-full

# With detailed FIT analysis
cycling garmin --analyze-fit

# Export to CSV
cycling garmin --export-csv activities.csv

# Dry run (test without saving)
cycling garmin --dry-run
```

#### From Intervals.icu:

```bash
# Incremental sync (new activities only)
cycling sync

# Full sync (last 365 days)
cycling sync --force-full

# Export to CSV
cycling sync --export-csv activities.csv
```

### Machine Learning

```bash
# Train ML models on your data
cycling train

# Generate performance predictions
cycling predict

# View predictions with options
cycling predict --days 60 --save-plots
```

### View Analytics

```bash
# Launch web dashboard
cycling dashboard

# Generate Excel report
cycling report

# Generate report with custom filename
cycling report --output my_report.xlsx

# Generate CSV reports instead of Excel
cycling report --format csv
```

### Python API

```python
from src.integrations.intervals_icu import IntervalsICUClient
from src.storage.database.manager import DatabaseManager
from src.analytics.performance import PerformanceAnalyzer

# Fetch activities
client = IntervalsICUClient(athlete_id, api_key)
activities = client.get_activities()

# Analyze performance
db = DatabaseManager("data/silver/cycling_analytics.db")
analyzer = PerformanceAnalyzer(db.get_activities())
ftp_progression = analyzer.calculate_ftp_progression()
```

## Configuration

1. Copy the example configuration file:

```bash
cp .env.example .env
```

2. Edit `.env` and add your credentials:
   - **Intervals.icu**: Get API key from Settings → Developer Settings
   - **Garmin Connect**: Use your login credentials (stored locally, OAuth tokens cached after first login)

**Security Note**: Never commit `.env` files to version control. The app caches Garmin OAuth tokens after first login to avoid storing passwords.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src
```

## Requirements

- Python 3.11+
- 2GB RAM minimum
- 500MB disk space for data storage

## License

MIT
