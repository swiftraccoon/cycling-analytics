"""Command-line interface for the cycling analytics platform."""

import sys
import subprocess
import argparse
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='cycling',
        description='Cycling Analytics Platform - Track, analyze, and improve your cycling performance',
        epilog='For more information on a specific command, run: cycling <command> --help'
    )
    
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='<command>'
    )
    
    # Sync command (Intervals.icu)
    sync_parser = subparsers.add_parser(
        'sync',
        help='Sync activities from Intervals.icu API',
        description='Synchronize your cycling activities from Intervals.icu to the local database'
    )
    sync_parser.add_argument(
        '--days',
        type=int,
        help='Number of days to sync (default: 30)'
    )
    sync_parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-sync of all activities'
    )
    
    # Garmin command
    garmin_parser = subparsers.add_parser(
        'garmin',
        help='Sync activities from Garmin Connect with FIT files',
        description='Synchronize activities from Garmin Connect and download FIT files for detailed analysis'
    )
    garmin_parser.add_argument(
        '--email',
        help='Garmin Connect email (or set GARMIN_EMAIL env var)'
    )
    garmin_parser.add_argument(
        '--password',
        help='Garmin Connect password (or set GARMIN_PASSWORD env var)'
    )
    garmin_parser.add_argument(
        '--start-date',
        help='Start date for sync (YYYY-MM-DD)'
    )
    garmin_parser.add_argument(
        '--end-date',
        help='End date for sync (YYYY-MM-DD)'
    )
    garmin_parser.add_argument(
        '--force-full',
        action='store_true',
        help='Force full re-sync, ignoring last sync date'
    )
    garmin_parser.add_argument(
        '--no-fit',
        action='store_true',
        help='Skip downloading FIT files'
    )
    garmin_parser.add_argument(
        '--analyze-fit',
        action='store_true',
        help='Analyze FIT files after download'
    )
    garmin_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be synced without saving'
    )
    garmin_parser.add_argument(
        '--export-csv',
        help='Export fetched activities to CSV file'
    )
    garmin_parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of activities to sync'
    )
    
    # Ingest command
    ingest_parser = subparsers.add_parser(
        'ingest',
        help='Process CSV files from data/bronze/incoming',
        description='Ingest and process CSV files containing cycling activity data'
    )
    ingest_parser.add_argument(
        '--file',
        help='Specific CSV file to ingest'
    )
    ingest_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview ingestion without saving to database'
    )
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser(
        'dashboard',
        help='Launch the Streamlit dashboard',
        description='Launch the interactive web dashboard for visualizing your cycling data'
    )
    dashboard_parser.add_argument(
        '--port',
        type=int,
        default=8502,
        help='Port to run the dashboard on (default: 8502)'
    )
    dashboard_parser.add_argument(
        '--host',
        default='localhost',
        help='Host to run the dashboard on (default: localhost)'
    )
    
    # Report command
    report_parser = subparsers.add_parser(
        'report',
        help='Generate Excel report',
        description='Generate a comprehensive Excel report of your cycling activities'
    )
    report_parser.add_argument(
        '--output',
        help='Output file path (default: reports/cycling_report_YYYYMMDD.xlsx)'
    )
    report_parser.add_argument(
        '--start-date',
        help='Start date for report (YYYY-MM-DD)'
    )
    report_parser.add_argument(
        '--end-date',
        help='End date for report (YYYY-MM-DD)'
    )
    report_parser.add_argument(
        '--activity-type',
        help='Filter by activity type (e.g., Ride, Run)'
    )
    
    # Train command
    train_parser = subparsers.add_parser(
        'train',
        help='Train ML models on your cycling data',
        description='Train machine learning models for performance prediction and analysis'
    )
    train_parser.add_argument(
        '--source',
        choices=['intervals', 'garmin', 'auto'],
        default='auto',
        help='Data source for training (default: auto-detect)'
    )
    train_parser.add_argument(
        '--model-type',
        choices=['ftp', 'performance', 'all'],
        default='all',
        help='Type of model to train (default: all)'
    )
    train_parser.add_argument(
        '--output-dir',
        help='Directory to save trained models (default: data/models)'
    )
    
    # Predict command
    predict_parser = subparsers.add_parser(
        'predict',
        help='Generate performance predictions',
        description='Generate predictions using trained machine learning models'
    )
    predict_parser.add_argument(
        '--days-ahead',
        type=int,
        default=30,
        help='Number of days to predict ahead (default: 30)'
    )
    predict_parser.add_argument(
        '--metric',
        choices=['ftp', 'performance', 'readiness', 'all'],
        default='all',
        help='Metric to predict (default: all)'
    )
    predict_parser.add_argument(
        '--output',
        help='Output file for predictions (optional)'
    )
    
    # Quality command
    quality_parser = subparsers.add_parser(
        'quality',
        help='Check data quality and integrity',
        description='Analyze data quality, completeness, and identify issues'
    )
    quality_parser.add_argument(
        '--fix',
        action='store_true',
        help='Attempt to fix identified issues automatically'
    )
    quality_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed quality metrics'
    )
    
    # Reprocess command
    reprocess_parser = subparsers.add_parser(
        'reprocess',
        help='Reprocess existing FIT files',
        description='Reprocess downloaded FIT files to extract all available data'
    )
    reprocess_parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing even if FIT data already exists'
    )
    reprocess_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be processed without updating database'
    )
    reprocess_parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify FIT data extraction after processing'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command is None:
        parser.print_help()
        return
    
    # Route to appropriate handler
    if args.command == 'sync':
        run_sync(args)
    elif args.command == 'garmin':
        run_garmin(args)
    elif args.command == 'ingest':
        run_ingest(args)
    elif args.command == 'dashboard':
        run_dashboard(args)
    elif args.command == 'report':
        run_report(args)
    elif args.command == 'train':
        run_train(args)
    elif args.command == 'predict':
        run_predict(args)
    elif args.command == 'quality':
        cmd_quality(args)
    elif args.command == 'reprocess':
        cmd_reprocess(args)


def run_sync(args):
    """Run Intervals.icu sync."""
    script_path = Path(__file__).parent / "integrations" / "intervals_sync.py"
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)
    
    # Build command with arguments
    cmd = [sys.executable, str(script_path)]
    if hasattr(args, 'days') and args.days:
        cmd.extend(['--days', str(args.days)])
    if hasattr(args, 'force') and args.force:
        cmd.append('--force')
    
    subprocess.run(cmd)


def run_garmin(args):
    """Run Garmin Connect sync."""
    script_path = Path(__file__).parent / "integrations" / "garmin_sync.py"
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)
    
    # Build command with arguments
    cmd = [sys.executable, str(script_path)]
    
    # Add all provided arguments
    if hasattr(args, 'email') and args.email:
        cmd.extend(['--email', args.email])
    if hasattr(args, 'password') and args.password:
        cmd.extend(['--password', args.password])
    if hasattr(args, 'start_date') and args.start_date:
        cmd.extend(['--start-date', args.start_date])
    if hasattr(args, 'end_date') and args.end_date:
        cmd.extend(['--end-date', args.end_date])
    if hasattr(args, 'force_full') and args.force_full:
        cmd.append('--force-full')
    if hasattr(args, 'no_fit') and args.no_fit:
        cmd.append('--no-fit')
    if hasattr(args, 'analyze_fit') and args.analyze_fit:
        cmd.append('--analyze-fit')
    if hasattr(args, 'dry_run') and args.dry_run:
        cmd.append('--dry-run')
    if hasattr(args, 'export_csv') and args.export_csv:
        cmd.extend(['--export-csv', args.export_csv])
    if hasattr(args, 'limit') and args.limit:
        cmd.extend(['--limit', str(args.limit)])
    
    subprocess.run(cmd)


def run_ingest(args):
    """Run CSV ingestion."""
    script_path = Path(__file__).parent / "storage" / "process_csv.py"
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)
    
    # Build command with arguments
    cmd = [sys.executable, str(script_path)]
    if hasattr(args, 'file') and args.file:
        cmd.extend(['--file', args.file])
    if hasattr(args, 'dry_run') and args.dry_run:
        cmd.append('--dry-run')
    
    subprocess.run(cmd)


def run_dashboard(args):
    """Launch Streamlit dashboard."""
    app_path = Path(__file__).parent / "app" / "dashboard.py"
    if not app_path.exists():
        print(f"Error: Dashboard not found at {app_path}")
        sys.exit(1)
    
    # Build streamlit command
    cmd = [
        "streamlit", "run", str(app_path),
        "--server.port", str(args.port),
        "--server.address", args.host
    ]
    
    subprocess.run(cmd)


def run_report(args):
    """Generate Excel report."""
    script_path = Path(__file__).parent / "reports" / "generate_excel.py"
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)
    
    # Build command with arguments
    cmd = [sys.executable, str(script_path)]
    if hasattr(args, 'output') and args.output:
        cmd.extend(['--output', args.output])
    if hasattr(args, 'start_date') and args.start_date:
        cmd.extend(['--start-date', args.start_date])
    if hasattr(args, 'end_date') and args.end_date:
        cmd.extend(['--end-date', args.end_date])
    if hasattr(args, 'activity_type') and args.activity_type:
        cmd.extend(['--activity-type', args.activity_type])
    
    subprocess.run(cmd)


def run_train(args):
    """Train ML models."""
    # Determine which training script to use
    source = args.source if hasattr(args, 'source') else 'auto'
    
    if source == 'auto':
        # Try to detect which data source has more data
        from src.storage.database.manager import DatabaseManager
        db = DatabaseManager()
        activities = db.get_activities()
        
        has_ftp = False
        has_power = False
        
        if not activities.is_empty():
            if 'icu_ftp' in activities.columns:
                has_ftp = activities['icu_ftp'].is_not_null().sum() > 0
            if 'normalized_power' in activities.columns:
                has_power = activities['normalized_power'].is_not_null().sum() > 0
        
        if has_ftp:
            script_name = "train_intervals.py"
            print("Using Intervals.icu FTP data for training...")
        elif has_power:
            script_name = "train_garmin.py"
            print("Using Garmin power data for training...")
        else:
            print("Error: No power or FTP data found. Sync activities with power data first.")
            sys.exit(1)
    elif source == 'intervals':
        script_name = "train_intervals.py"
    elif source == 'garmin':
        script_name = "train_garmin.py"
    else:
        script_name = "train_intervals.py"
    
    script_path = Path(__file__).parent / "ml" / script_name
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)
    
    # Build command with arguments
    cmd = [sys.executable, str(script_path)]
    
    # The training scripts don't currently accept arguments, but we could add them
    subprocess.run(cmd)


def run_predict(args):
    """Generate predictions."""
    script_path = Path(__file__).parent / "ml" / "generate_predictions.py"
    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)
    
    # Build command with arguments
    cmd = [sys.executable, str(script_path)]
    if hasattr(args, 'days_ahead') and args.days_ahead:
        cmd.extend(['--days-ahead', str(args.days_ahead)])
    if hasattr(args, 'metric') and args.metric:
        cmd.extend(['--metric', args.metric])
    if hasattr(args, 'output') and args.output:
        cmd.extend(['--output', args.output])
    
    subprocess.run(cmd)


def cmd_reprocess(args):
    """Handle reprocess command."""
    from src.integrations.fit_extractor import DirectFITExtractor
    from src.storage.database.manager import DatabaseManager
    import polars as pl
    
    print("="*70)
    print("FIT FILE REPROCESSING")
    print("="*70)
    
    # Use the direct extractor that actually works
    extractor = DirectFITExtractor()
    fit_files = list(Path("data/bronze/fit_files").glob("*.fit"))
    
    if not fit_files:
        print("\nERROR: No FIT files found in data/bronze/fit_files/")
        print("Run 'cycling garmin' first to download FIT files")
        sys.exit(1)
    
    print(f"\nFound {len(fit_files)} FIT files to process")
    
    if args.dry_run:
        print("\nDRY RUN MODE - No database changes will be made")
        for f in fit_files[:5]:
            print(f"  Would process: {f.name}")
        if len(fit_files) > 5:
            print(f"  ... and {len(fit_files)-5} more")
        return
    
    # Extract all FIT data
    print("\nExtracting FIT data...")
    df = extractor.extract_all_fit_files()
    
    if df.is_empty():
        print("ERROR: Failed to extract any FIT data")
        sys.exit(1)
    
    # Save to database
    db = DatabaseManager()
    
    if not args.force:
        # Check if we already have data
        existing = db.get_activities()
        if not existing.is_empty() and 'threshold_power' in existing.columns:
            ftp_count = existing['threshold_power'].is_not_null().sum()
            if ftp_count > 0:
                print(f"\n✅ Database already has {ftp_count} activities with FTP data")
                print("Use --force to overwrite")
                return
    
    # Save directly without strict validation
    print("\nSaving to database...")
    import sqlite3
    conn = sqlite3.connect(db.db_path)
    df_pandas = df.to_pandas()
    df_pandas.to_sql('activities', conn, if_exists='replace', index=False)
    conn.close()
    
    results = {
        'total_files': len(fit_files),
        'processed': len(df),
        'updated': len(df),
        'errors': len(fit_files) - len(df)
    }
    
    # Show results
    print("\n" + "="*40)
    print("REPROCESSING RESULTS")
    print("="*40)
    print(f"Total FIT files: {results.get('total_files', 0)}")
    print(f"Successfully processed: {results.get('processed', 0)}")
    print(f"Database updated: {results.get('updated', 0)}")
    print(f"Errors: {results.get('errors', 0)}")
    
    # Verify if requested
    if args.verify:
        print("\nVerifying FIT data extraction...")
        report = extractor.get_fit_coverage_report(df)
        
        print("\n" + "="*40)
        print("FIT DATA COVERAGE")
        print("="*40)
        
        for field, stats in report.get('field_coverage', {}).items():
            status = "✅" if stats['percentage'] > 50 else "⚠️" if stats['percentage'] > 0 else "❌"
            print(f"{status} {field}: {stats['count']}/{report['total_activities']} ({stats['percentage']:.1f}%)")
        
        print(f"\nOverall FIT completeness: {report.get('overall_fit_coverage', 0):.1f}%")
        
        if report.get('overall_fit_coverage', 0) < 50:
            print("\n⚠️  WARNING: Low FIT data coverage detected")
            print("Consider running 'cycling reprocess --force' to reprocess all files")


def cmd_quality(args):
    """Handle quality command."""
    from src.storage.database.manager import DatabaseManager
    from src.data.validator import DataValidator
    import polars as pl
    
    print("="*70)
    print("DATA QUALITY AUDIT")
    print("="*70)
    
    db = DatabaseManager()
    activities = db.get_activities()
    
    if activities.is_empty():
        print("\nERROR: No data in database")
        sys.exit(1)
    
    print(f"\nAnalyzing {len(activities)} activities...\n")
    
    # Get quality metrics
    metrics = DataValidator.ensure_data_quality(activities)
    
    # Overall score
    score = metrics['quality_score']
    print(f"OVERALL DATA QUALITY SCORE: {score:.1f}%")
    
    if score >= 99:
        print("✅ EXCELLENT - Data quality meets highest standards")
    elif score >= 95:
        print("⚠️  ACCEPTABLE - Must reach 99% for production")
    else:
        print("❌ UNACCEPTABLE - Below 95% minimum threshold!")
    
    if args.verbose:
        print("\nDETAILED METRICS:")
        print("-"*40)
        
        # Field completeness
        print("\nField Completeness:")
        for field, null_pct in sorted(metrics['null_percentages'].items()):
            completeness = 100 - null_pct
            status = "✅" if completeness >= 90 else "⚠️" if completeness >= 70 else "❌"
            print(f"  {status} {field}: {completeness:.1f}% complete")
        
        # Value ranges
        if metrics.get('value_ranges'):
            print("\nValue Ranges:")
            for field, ranges in metrics['value_ranges'].items():
                if ranges:
                    print(f"  {field}: {ranges['min']:.2f} - {ranges['max']:.2f} (mean: {ranges['mean']:.2f})")
    
    if args.fix and score < 95:
        print("\nAttempting to fix data issues...")
        # Re-extract zone data
        print("- Extracting zone data from JSON fields...")
        activities = db._extract_zone_data(activities)
        
        # Re-validate
        print("- Re-validating data...")
        activities = DataValidator.validate_dataframe(activities)
        
        # Save fixed data
        print("- Saving corrected data...")
        db.save_activities(activities, update_existing=True)
        
        print("\n✅ Data fixes applied. Re-run quality check to verify.")


if __name__ == "__main__":
    main()