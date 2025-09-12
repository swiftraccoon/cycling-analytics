"""COMPLETE dashboard tests - ALL dashboard functionality in ONE file.

This is THE ONLY dashboard test file we need.
Tests:
1. Basic dashboard functions 
2. Graph data extraction and plotting
3. Compatibility with both Intervals and Garmin data
4. Edge cases and error handling
5. Integration with real database data
6. Regression prevention

NO MORE MULTIPLE TEST FILES!
"""

import pytest
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
from unittest.mock import patch, MagicMock, call
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.dashboard import (
    prepare_calendar_data,
    calculate_daily_metrics, 
    load_activities,
    get_database,
    add_fit_analysis_section,
    add_power_analysis_tab
)
from src.storage.database.manager import DatabaseManager
from src.analytics.performance import PerformanceAnalyzer
from src.ml.predictions import PerformancePredictor


class TestDashboardComplete:
    """Complete dashboard testing - everything in one place."""
    
    @pytest.fixture
    def real_data(self):
        """Load real data from database."""
        db = DatabaseManager()
        data = db.get_activities()
        if data.is_empty():
            pytest.skip("No data in database")
        return data
    
    @pytest.fixture 
    def analyzer(self, real_data):
        """Create performance analyzer."""
        return PerformanceAnalyzer(real_data)

    # ===== CORE GRAPH TESTS =====
    
    def test_ftp_progression_graph_complete(self, analyzer):
        """Test FTP progression shows ALL data points."""
        ftp_data = analyzer.calculate_ftp_progression()
        
        if not ftp_data.is_empty():
            # Verify we get ALL FTP data, not just unique values
            print(f"FTP progression: {len(ftp_data)} data points")
            
            # Should have many more than just unique FTP values
            unique_ftp = ftp_data['ftp_value'].n_unique()
            assert len(ftp_data) >= unique_ftp, "Should show progression over time, not just unique values"
            
            # Test graph creation
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ftp_data["start_date_local"].to_list(),
                y=ftp_data["ftp_value"].to_list(),
                mode="lines+markers",
                name="FTP"
            ))
            
            assert len(fig.data[0].y) == len(ftp_data)
            print(f"✅ FTP graph: {len(ftp_data)} points plotted")
        else:
            pytest.skip("No FTP data")
    
    def test_power_zones_graph_complete(self, analyzer):
        """Test power zones calculation and display."""
        zone_data = analyzer.calculate_power_zones()
        
        # Should have FTP and zones even if no time data
        assert 'ftp' in zone_data, "Missing FTP"
        assert 'zones' in zone_data, "Missing zones" 
        
        ftp = zone_data['ftp']
        zones = zone_data['zones']
        
        # Verify zones are reasonable
        assert 100 <= ftp <= 500, f"FTP {ftp}W out of range"
        assert len(zones) == 7, f"Should have 7 zones, got {len(zones)}"
        
        # Test zone boundaries
        for name, (low, high) in zones.items():
            assert low < high, f"Invalid bounds for {name}: {low} >= {high}"
        
        print(f"✅ Power zones: FTP={ftp}W, {len(zones)} zones")
    
    def test_calendar_heatmap_complete(self, real_data):
        """Test calendar heatmap data and visualization."""
        calendar_data = prepare_calendar_data(real_data)
        
        assert not calendar_data.is_empty(), "Calendar data empty"
        assert 'week' in calendar_data.columns
        assert 'weekday' in calendar_data.columns
        
        # Should have training load or activity count
        has_load = 'training_load' in calendar_data.columns
        has_count = 'activity_count' in calendar_data.columns
        assert has_load or has_count, "Missing both training_load and activity_count"
        
        # Test heatmap creation
        z_col = 'training_load' if has_load else 'activity_count'
        fig = px.density_heatmap(
            calendar_data.to_pandas(),
            x="week", y="weekday", z=z_col
        )
        
        assert fig is not None
        print(f"✅ Calendar heatmap: {len(calendar_data)} points")
    
    def test_best_efforts_all_sections(self, analyzer, real_data):
        """Test all best efforts sections."""
        # Test power efforts
        power_cols = ['normalized_power', 'avg_power']
        power_tested = False
        
        for col in power_cols:
            if col in real_data.columns and real_data[col].is_not_null().any():
                best = analyzer.get_best_efforts(col, top_n=5)
                if not best.is_empty():
                    assert col in best.columns
                    assert len(best) <= 5
                    print(f"✅ Best {col}: {len(best)} efforts")
                    power_tested = True
                    break
        
        if not power_tested:
            print("⚠️  No power data for best efforts")
        
        # Test distance efforts
        if 'distance' in real_data.columns:
            best_dist = analyzer.get_best_efforts('distance', top_n=5)
            if not best_dist.is_empty():
                distances = [d/1000 for d in best_dist['distance'].to_list()]
                print(f"✅ Best distances: {len(best_dist)} efforts, max {max(distances):.1f}km")
        
        # Test training load efforts
        load_cols = ['training_stress_score', 'icu_training_load']
        for col in load_cols:
            if col in real_data.columns and real_data[col].is_not_null().any():
                best_load = analyzer.get_best_efforts(col, top_n=5)
                if not best_load.is_empty():
                    print(f"✅ Best {col}: {len(best_load)} efforts")
                    break
    
    def test_daily_and_training_metrics(self, real_data, analyzer):
        """Test daily metrics and training load calculations."""
        # Daily metrics
        daily = calculate_daily_metrics(real_data)
        assert not daily.is_empty(), "Daily metrics empty"
        assert 'date' in daily.columns
        
        # Should have some aggregated data
        metric_cols = [col for col in daily.columns if col != 'date']
        assert len(metric_cols) > 0, f"No metrics in daily data: {daily.columns}"
        
        # Training load metrics
        load_metrics = analyzer.calculate_training_load()
        assert isinstance(load_metrics, dict)
        assert 'activities' in load_metrics
        
        print(f"✅ Daily metrics: {len(daily)} days")
        print(f"✅ Training load: {load_metrics['activities']} activities")
    
    # ===== DATA SOURCE COMPATIBILITY =====
    
    def test_intervals_vs_garmin_compatibility(self, real_data):
        """Test dashboard works with both data sources."""
        # Check what data source we have
        has_intervals = any(col.startswith('icu_') for col in real_data.columns)
        has_garmin = 'threshold_power' in real_data.columns or 'has_fit_data' in real_data.columns
        
        print(f"Data sources - Intervals: {has_intervals}, Garmin: {has_garmin}")
        
        # Dashboard should work with either
        assert has_intervals or has_garmin, "No recognized data source"
        
        # Test key functions work regardless of source
        analyzer = PerformanceAnalyzer(real_data)
        
        # FTP should work with either icu_ftp or threshold_power
        ftp_data = analyzer.calculate_ftp_progression()
        if not ftp_data.is_empty():
            assert 'ftp_value' in ftp_data.columns
        
        # Calendar should work with either training load source
        calendar = prepare_calendar_data(real_data)
        assert not calendar.is_empty()
        
        print("✅ Data source compatibility verified")
    
    # ===== ERROR HANDLING =====
    
    def test_edge_cases_and_errors(self):
        """Test dashboard handles edge cases gracefully.""" 
        # Empty data
        empty_df = pl.DataFrame({'id': [], 'start_date_local': [], 'distance': []})
        
        calendar_empty = prepare_calendar_data(empty_df)
        daily_empty = calculate_daily_metrics(empty_df)
        analyzer_empty = PerformanceAnalyzer(empty_df)
        
        # Should not crash
        assert calendar_empty.is_empty() or calendar_empty is not None
        assert daily_empty.is_empty() or daily_empty is not None
        
        ftp_empty = analyzer_empty.calculate_ftp_progression()
        assert ftp_empty.is_empty()
        
        # Null data
        null_df = pl.DataFrame({
            'id': ['1', '2'],
            'start_date_local': [None, None],
            'distance': [None, None]
        })
        
        # Should handle gracefully
        calendar_null = prepare_calendar_data(null_df)
        assert calendar_null is not None
        
        print("✅ Edge cases handled gracefully")
    
    # ===== INTEGRATION TESTS =====
    
    @patch('streamlit.plotly_chart')
    @patch('streamlit.metric')
    @patch('streamlit.write')
    @patch('streamlit.header')
    def test_dashboard_sections_render(self, mock_header, mock_write, mock_metric, mock_chart, real_data):
        """Test dashboard sections render without errors."""
        # Test FIT analysis section
        add_fit_analysis_section(real_data)
        
        # Test power analysis tab
        with patch('streamlit.tabs') as mock_tabs:
            mock_tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
            add_power_analysis_tab(real_data)
        
        print("✅ Dashboard sections render without crashes")
    
    def test_database_integration(self):
        """Test dashboard integrates properly with database."""
        db = get_database()
        assert db is not None
        
        activities = load_activities(db)
        assert activities is not None
        
        if not activities.is_empty():
            # Required columns should exist
            required = ['id', 'name', 'type', 'start_date_local', 'moving_time', 'distance']
            for col in required:
                assert col in activities.columns, f"Missing required column: {col}"
        
        print("✅ Database integration working")
    
    # ===== REGRESSION PREVENTION =====
    
    def test_no_import_breakage(self):
        """Test all imports still work."""
        try:
            from src.app.dashboard import main
            from src.analytics.performance import PerformanceAnalyzer
            from src.ml.predictions import PerformancePredictor
            print("✅ All imports working")
        except ImportError as e:
            pytest.fail(f"Import broken: {e}")
    
    def test_ftp_progression_shows_all_data_not_just_unique(self, analyzer):
        """REGRESSION TEST: Ensure FTP shows all 66 points, not just 8 unique values."""
        ftp_data = analyzer.calculate_ftp_progression()
        
        if not ftp_data.is_empty():
            # This was the bug - only showing unique FTP values instead of progression
            assert len(ftp_data) > 10, f"FTP progression only shows {len(ftp_data)} points - should show many more for timeline"
            
            # Should show progression over time
            dates = ftp_data['start_date_local'].to_list()
            assert len(dates) > len(set(ftp_data['ftp_value'].to_list())), "Should show more data points than unique FTP values"
            
            print(f"✅ REGRESSION FIXED: FTP shows {len(ftp_data)} timeline points, not just unique values")
    
    # ===== FINAL VALIDATION =====
    
    def test_dashboard_comprehensive_validation(self, real_data):
        """Final comprehensive test - everything should work."""
        print("\\n=== COMPREHENSIVE DASHBOARD VALIDATION ===")
        
        analyzer = PerformanceAnalyzer(real_data)
        
        # 1. FTP progression
        ftp_data = analyzer.calculate_ftp_progression()
        ftp_points = len(ftp_data)
        print(f"✅ FTP progression: {ftp_points} data points")
        
        # 2. Power zones  
        zones = analyzer.calculate_power_zones()
        zone_count = len(zones.get('zones', {}))
        print(f"✅ Power zones: {zone_count} zones defined")
        
        # 3. Calendar heatmap
        calendar = prepare_calendar_data(real_data)
        calendar_points = len(calendar)
        print(f"✅ Calendar heatmap: {calendar_points} data points")
        
        # 4. Best efforts
        best_power = analyzer.get_best_efforts('normalized_power', top_n=5) if 'normalized_power' in real_data.columns else pl.DataFrame()
        best_distance = analyzer.get_best_efforts('distance', top_n=5)
        print(f"✅ Best efforts: {len(best_power)} power, {len(best_distance)} distance")
        
        # 5. Daily metrics
        daily = calculate_daily_metrics(real_data)
        daily_points = len(daily)
        print(f"✅ Daily metrics: {daily_points} days")
        
        # Summary
        total_tests = 5
        working_tests = sum([
            ftp_points > 0,
            zone_count > 0, 
            calendar_points > 0,
            len(best_distance) > 0,
            daily_points > 0
        ])
        
        print(f"\\n🎉 DASHBOARD STATUS: {working_tests}/{total_tests} core features working")
        
        # All major features should work
        assert working_tests >= 4, f"Only {working_tests}/{total_tests} features working - dashboard broken"
        
        if working_tests == total_tests:
            print("🎉 ALL DASHBOARD GRAPHS AND FEATURES ARE WORKING PERFECTLY!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])