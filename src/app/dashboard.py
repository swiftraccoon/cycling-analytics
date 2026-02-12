"""Streamlit dashboard for cycling analytics."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import polars as pl

from src.storage.database.manager import DatabaseManager
from src.analytics.performance import PerformanceAnalyzer
from src.analytics.advanced import AdvancedAnalytics
from src.ml.predictions import PerformancePredictor
from src.ml.fit_feature_extractor import FITFeatureExtractor


st.set_page_config(
    page_title="Cycling Analytics Dashboard",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_database():
    """Get database connection."""
    return DatabaseManager()  # Uses default from config


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_activities(_db):
    """Load activities from database."""
    return _db.get_activities()


def main():
    """Main dashboard application."""
    st.title("🚴 Cycling Analytics Dashboard")
    
    # Load data
    db = get_database()
    activities_df = load_activities(db)
    
    if activities_df.is_empty():
        st.warning("No activities found in database. Please run 'cycling ingest' first.")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range filter
    min_date = activities_df["start_date_local"].min()
    max_date = activities_df["start_date_local"].max()
    
    # Quick date range selector
    date_option = st.sidebar.selectbox(
        "Quick Select",
        ["Last 30 days", "Last 90 days", "Last 6 months", "Last year", "All time", "Custom"]
    )
    
    # Calculate date range based on selection
    if date_option == "Last 30 days":
        default_start = max_date - timedelta(days=30)
        default_end = max_date
    elif date_option == "Last 90 days":
        default_start = max_date - timedelta(days=90)
        default_end = max_date
    elif date_option == "Last 6 months":
        default_start = max_date - timedelta(days=180)
        default_end = max_date
    elif date_option == "Last year":
        default_start = max_date - timedelta(days=365)
        default_end = max_date
    elif date_option == "All time":
        default_start = min_date
        default_end = max_date
    else:  # Custom
        default_start = max_date - timedelta(days=90)
        default_end = max_date
    
    # Convert to date objects for the widget
    if hasattr(default_start, 'date'):
        default_start = default_start.date()
    if hasattr(default_end, 'date'):
        default_end = default_end.date()
    if hasattr(min_date, 'date'):
        min_date = min_date.date()
    if hasattr(max_date, 'date'):
        max_date = max_date.date()
    
    # Show date input only for custom selection
    if date_option == "Custom":
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(default_start, default_end),
            min_value=min_date,
            max_value=max_date,
        )
    else:
        date_range = (default_start, default_end)
        st.sidebar.info(f"{default_start} to {default_end}")
    
    # Activity type filter
    activity_types = activities_df["type"].unique().to_list()
    selected_types = st.sidebar.multiselect(
        "Activity Types",
        options=activity_types,
        default=activity_types,
    )
    
    # Filter data - handle both single date and date range
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        # Single date selected
        start_date = date_range if not isinstance(date_range, tuple) else date_range[0]
        end_date = start_date
    
    # Convert dates to datetime for comparison with datetime column
    from datetime import datetime as dt
    start_datetime = dt.combine(start_date, dt.min.time())
    end_datetime = dt.combine(end_date, dt.max.time())
    
    filtered_df = activities_df.filter(
        (pl.col("start_date_local") >= start_datetime) &
        (pl.col("start_date_local") <= end_datetime) &
        (pl.col("type").is_in(selected_types))
    )
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer(filtered_df)
    
    # Overview metrics
    st.header("Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Activities", len(filtered_df))
    
    with col2:
        if "distance" in filtered_df.columns:
            # Convert to numeric if it's stored as string
            try:
                dist_col = filtered_df["distance"]
                if dist_col.dtype == pl.Utf8:  # String type
                    dist_col = dist_col.cast(pl.Float64, strict=False)
                total_distance = dist_col.sum()
                total_distance = total_distance / 1000 if total_distance is not None else 0
            except:
                total_distance = 0
        else:
            total_distance = 0
        st.metric("Total Distance", f"{total_distance:,.0f} km")
    
    with col3:
        if "moving_time" in filtered_df.columns:
            # Convert to numeric if it's stored as string
            try:
                time_col = filtered_df["moving_time"]
                if time_col.dtype == pl.Utf8:  # String type
                    time_col = time_col.cast(pl.Float64, strict=False)
                total_time = time_col.sum()
                total_time = total_time / 3600 if total_time is not None else 0
            except:
                total_time = 0
        else:
            total_time = 0
        st.metric("Total Time", f"{total_time:,.0f} hrs")
    
    with col4:
        if "total_elevation_gain" in filtered_df.columns:
            # Convert to numeric if it's stored as string
            try:
                elev_col = filtered_df["total_elevation_gain"]
                if elev_col.dtype == pl.Utf8:  # String type
                    elev_col = elev_col.cast(pl.Float64, strict=False)
                total_elevation = elev_col.sum()
                total_elevation = total_elevation if total_elevation is not None else 0
            except:
                total_elevation = 0
        else:
            total_elevation = 0
        st.metric("Total Elevation", f"{total_elevation:,.0f} m")
    
    with col5:
        if "distance" in filtered_df.columns:
            # Convert to numeric if it's stored as string
            try:
                dist_col = filtered_df["distance"]
                if dist_col.dtype == pl.Utf8:  # String type
                    dist_col = dist_col.cast(pl.Float64, strict=False)
                avg_distance = dist_col.mean()
                avg_distance = avg_distance / 1000 if avg_distance is not None else 0
            except:
                avg_distance = 0
        else:
            avg_distance = 0
        st.metric("Avg Distance", f"{avg_distance:.1f} km")
    
    # Training Load Section
    st.header("Training Load")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Calculate daily metrics
        daily_metrics = calculate_daily_metrics(filtered_df)
        
        if not daily_metrics.is_empty():
            # Create training load chart
            fig = go.Figure()
            
            # Add CTL (Fitness)
            if "icu_fitness" in daily_metrics.columns:
                fig.add_trace(go.Scatter(
                    x=daily_metrics["date"],
                    y=daily_metrics["icu_fitness"],
                    mode="lines",
                    name="CTL (Fitness)",
                    line=dict(color="blue", width=2),
                ))
            
            # Add ATL (Fatigue)
            if "icu_fatigue" in daily_metrics.columns:
                fig.add_trace(go.Scatter(
                    x=daily_metrics["date"],
                    y=daily_metrics["icu_fatigue"],
                    mode="lines",
                    name="ATL (Fatigue)",
                    line=dict(color="red", width=2),
                ))
            
            # Add TSB (Form)
            if "icu_fitness" in daily_metrics.columns and "icu_fatigue" in daily_metrics.columns:
                # Convert to numeric if stored as strings
                fitness = daily_metrics["icu_fitness"]
                fatigue = daily_metrics["icu_fatigue"]
                
                if fitness.dtype == pl.Utf8:
                    fitness = fitness.cast(pl.Float64, strict=False)
                if fatigue.dtype == pl.Utf8:
                    fatigue = fatigue.cast(pl.Float64, strict=False)
                
                tsb = fitness - fatigue
                fig.add_trace(go.Scatter(
                    x=daily_metrics["date"],
                    y=tsb,
                    mode="lines",
                    name="TSB (Form)",
                    line=dict(color="green", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0,255,0,0.1)",
                ))
            
            fig.update_layout(
                title="Training Load Progression",
                xaxis_title="Date",
                yaxis_title="Training Load",
                hovermode="x unified",
                height=400,
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Current training load metrics
        training_load = analyzer.calculate_training_load(days=42)
        
        st.subheader("Current Metrics")
        if training_load.get("ctl"):
            st.metric("CTL (Fitness)", f"{training_load['ctl']:.1f}")
        if training_load.get("atl"):
            st.metric("ATL (Fatigue)", f"{training_load['atl']:.1f}")
        if training_load.get("tsb"):
            tsb_value = training_load['tsb']
            delta_color = "normal" if tsb_value > -10 else "inverse"
            st.metric("TSB (Form)", f"{tsb_value:.1f}", delta_color=delta_color)
    
    # Weekly Summary
    st.header("Weekly Summary")
    
    weekly_summary = analyzer.calculate_weekly_summary(weeks=12)
    
    if not weekly_summary.is_empty() and "week" in weekly_summary.columns:
        # Create weekly volume chart
        fig = go.Figure()
        
        if "total_hours" in weekly_summary.columns:
            fig.add_trace(go.Bar(
                x=weekly_summary["week"],
                y=weekly_summary["total_hours"],
                name="Hours",
                marker_color="lightblue",
                yaxis="y",
            ))
        
        if "total_load" in weekly_summary.columns:
            fig.add_trace(go.Scatter(
                x=weekly_summary["week"],
                y=weekly_summary["total_load"],
                name="Training Load",
                line=dict(color="red", width=2),
                yaxis="y2",
            ))
        
        fig.update_layout(
            title="Weekly Training Volume",
            xaxis_title="Week",
            yaxis=dict(title="Hours", side="left"),
            yaxis2=dict(title="Training Load", overlaying="y", side="right"),
            hovermode="x unified",
            height=400,
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Power Analysis
    st.header("Power Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # FTP Progression
        ftp_data = analyzer.calculate_ftp_progression()
        
        if not ftp_data.is_empty():
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=ftp_data["start_date_local"],
                y=ftp_data["ftp_value"],
                mode="lines+markers",
                name="FTP",
                line=dict(color="purple", width=2),
                marker=dict(size=8),
            ))
            
            if "icu_eftp" in ftp_data.columns:
                fig.add_trace(go.Scatter(
                    x=ftp_data["start_date_local"],
                    y=ftp_data["icu_eftp"],
                    mode="lines+markers",
                    name="eFTP",
                    line=dict(color="orange", width=2, dash="dash"),
                    marker=dict(size=6),
                ))
            
            fig.update_layout(
                title="FTP Progression",
                xaxis_title="Date",
                yaxis_title="Power (W)",
                hovermode="x unified",
                height=350,
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Power zones distribution
        zone_data = analyzer.calculate_power_zones()
        
        if zone_data.get("zone_times_hours"):
            zones = []
            hours = []
            
            zone_names = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7"]
            for i, zone in enumerate(zone_names):
                key = f"z{i+1}_secs"
                if key in zone_data["zone_times_hours"]:
                    zones.append(zone)
                    hours.append(zone_data["zone_times_hours"][key])
            
            if zones:
                fig = go.Figure(data=[
                    go.Bar(x=zones, y=hours, marker_color=px.colors.sequential.Viridis)
                ])
                
                fig.update_layout(
                    title=f"Power Zone Distribution (FTP: {zone_data.get('ftp', 'N/A')}W)",
                    xaxis_title="Zone",
                    yaxis_title="Hours",
                    height=350,
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Best Efforts
    st.header("Best Efforts")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Best Power (NP)")
        # Try normalized power from either source
        power_col = None
        if "icu_normalized_watts" in filtered_df.columns and filtered_df["icu_normalized_watts"].is_not_null().any():
            power_col = "icu_normalized_watts"
        elif "normalized_power" in filtered_df.columns and filtered_df["normalized_power"].is_not_null().any():
            power_col = "normalized_power"
        elif "avg_power" in filtered_df.columns and filtered_df["avg_power"].is_not_null().any():
            power_col = "avg_power"
        
        if power_col:
            best_power = analyzer.get_best_efforts(power_col, top_n=5)
            if not best_power.is_empty():
                for row in best_power.to_dicts():
                    st.write(f"**{row[power_col]:.0f}W** - {row['name'][:25]} ({row['start_date_local'].strftime('%Y-%m-%d')})")
    
    with col2:
        st.subheader("Longest Rides")
        best_distance = analyzer.get_best_efforts("distance", top_n=5)
        if not best_distance.is_empty():
            for row in best_distance.to_dicts():
                st.write(f"**{row['distance']/1000:.1f}km** - {row['name'][:25]} ({row['start_date_local'].strftime('%Y-%m-%d')})")
    
    with col3:
        st.subheader("Highest Training Load")
        # Try training load from either source
        load_col = None
        if "icu_training_load" in filtered_df.columns and filtered_df["icu_training_load"].is_not_null().any():
            load_col = "icu_training_load"
        elif "training_stress_score" in filtered_df.columns and filtered_df["training_stress_score"].is_not_null().any():
            load_col = "training_stress_score"
        
        if load_col:
            best_load = analyzer.get_best_efforts(load_col, top_n=5)
            if not best_load.is_empty():
                for row in best_load.to_dicts():
                    st.write(f"**{row[load_col]:.0f}** - {row['name'][:25]} ({row['start_date_local'].strftime('%Y-%m-%d')})")
    
    # Activity Calendar Heatmap
    st.header("Activity Calendar")
    
    # Prepare calendar data
    calendar_data = prepare_calendar_data(filtered_df)
    
    if not calendar_data.is_empty():
        fig = px.density_heatmap(
            calendar_data.to_pandas(),
            x="week",
            y="weekday",
            z="training_load",
            color_continuous_scale="Viridis",
            labels={"training_load": "Load"},
            title="Training Load Heatmap",
        )
        
        fig.update_layout(
            xaxis_title="Week of Year",
            yaxis_title="Day of Week",
            height=300,
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ML Predictions Section
    st.header("🤖 Machine Learning Predictions")
    
    # Initialize ML predictor
    ml_predictor = PerformancePredictor(filtered_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("FTP Progression Forecast")
        
        # Predict FTP progression
        ftp_prediction = ml_predictor.predict_ftp_progression(days_ahead=30)
        
        if "error" not in ftp_prediction:
            # Create FTP prediction chart
            fig = go.Figure()
            
            # Add predicted FTP line
            fig.add_trace(go.Scatter(
                x=ftp_prediction["predictions"]["dates"],
                y=ftp_prediction["predictions"]["ftp_values"],
                mode="lines",
                name="Predicted FTP",
                line=dict(color="purple", width=3),
            ))
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=ftp_prediction["predictions"]["dates"],
                y=ftp_prediction["predictions"]["confidence_upper"],
                mode="lines",
                name="Upper bound",
                line=dict(color="purple", width=1, dash="dash"),
                showlegend=False,
            ))
            
            fig.add_trace(go.Scatter(
                x=ftp_prediction["predictions"]["dates"],
                y=ftp_prediction["predictions"]["confidence_lower"],
                mode="lines",
                name="Lower bound",
                line=dict(color="purple", width=1, dash="dash"),
                fill="tonexty",
                fillcolor="rgba(128,0,128,0.1)",
                showlegend=False,
            ))
            
            # Add current FTP marker
            fig.add_trace(go.Scatter(
                x=[ftp_prediction["predictions"]["dates"][0]],
                y=[ftp_prediction["current_ftp"]],
                mode="markers",
                name="Current FTP",
                marker=dict(size=10, color="red"),
            ))
            
            fig.update_layout(
                title=f"30-Day FTP Forecast (Current: {ftp_prediction['current_ftp']:.0f}W)",
                xaxis_title="Date",
                yaxis_title="FTP (Watts)",
                height=350,
                hovermode="x unified",
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics
            col1_1, col1_2, col1_3 = st.columns(3)
            with col1_1:
                st.metric(
                    "Current FTP",
                    f"{ftp_prediction['current_ftp']:.0f}W"
                )
            with col1_2:
                st.metric(
                    "Predicted (30d)",
                    f"{ftp_prediction['predictions']['ftp_values'][-1]:.0f}W",
                    delta=f"+{ftp_prediction['expected_gain']:.0f}W"
                )
            with col1_3:
                st.metric(
                    "Best Model",
                    ftp_prediction['best_model'].upper(),
                    delta=f"R²: {ftp_prediction['model_scores'][ftp_prediction['best_model']]:.2f}"
                )
        else:
            st.info(f"FTP prediction unavailable: {ftp_prediction.get('error', 'Insufficient data')}")
            st.caption(f"Need at least {ftp_prediction.get('required_activities', 10)} activities with FTP data")
    
    with col2:
        st.subheader("Performance Readiness")
        
        # Get readiness assessment
        readiness = ml_predictor.predict_performance_readiness()
        
        if "error" not in readiness:
            # Create readiness gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=readiness["overall_readiness"],
                title={'text': "Overall Readiness"},
                delta={'reference': 70, 'relative': False},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display readiness factors
            if "readiness_factors" in readiness:
                st.write("**Readiness Factors:**")
                for factor, score in readiness["readiness_factors"].items():
                    progress_color = "normal" if score >= 70 else "off"
                    st.progress(score/100, text=f"{factor.capitalize()}: {score:.0f}%")
            
            # Display recommendations
            if readiness.get("recommendations"):
                st.write("**Recommendations:**")
                for rec in readiness["recommendations"]:
                    st.write(f"• {rec}")
        else:
            st.info("Readiness assessment unavailable")
    
    # FIT Data Analysis Section
    add_fit_analysis_section(filtered_df)
    
    # Season Trajectory
    st.subheader("Season Trajectory Forecast")
    
    season_forecast = ml_predictor.forecast_season_trajectory(months_ahead=3)
    
    if "error" not in season_forecast and "forecasts" in season_forecast:
        # Create tabs for different metrics
        metric_tabs = st.tabs(list(season_forecast["forecasts"].keys()))
        
        for tab, (metric, forecast) in zip(metric_tabs, season_forecast["forecasts"].items()):
            with tab:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Trend",
                        forecast["trend"].upper(),
                        delta=f"{forecast['monthly_change']:.1f}/month"
                    )
                
                with col2:
                    st.metric(
                        "Model Confidence",
                        f"{forecast['r2_score']:.2%}"
                    )
                
                with col3:
                    if forecast["predictions"]:
                        st.metric(
                            "3-Month Forecast",
                            f"{forecast['predictions'][-1]:.0f}",
                            delta=f"{forecast['predictions'][-1] - forecast['predictions'][0]:.0f}"
                        )
                
                # Create forecast chart
                if forecast["predictions"]:
                    months = [f"Month +{i+1}" for i in range(len(forecast["predictions"]))]
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=months,
                        y=forecast["predictions"],
                        mode="lines+markers",
                        name=metric.replace("_", " ").title(),
                        line=dict(width=2),
                        marker=dict(size=8),
                    ))
                    
                    fig.update_layout(
                        xaxis_title="Future Month",
                        yaxis_title=metric.replace("_", " ").title(),
                        height=250,
                        showlegend=False,
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Display season recommendations
        if season_forecast.get("recommendations"):
            st.write("**Season Planning Recommendations:**")
            for rec in season_forecast["recommendations"]:
                st.write(f"• {rec}")
    else:
        st.info("Season forecast requires at least 3 months of historical data")


def calculate_daily_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate daily aggregated metrics."""
    if df.is_empty():
        return pl.DataFrame()
    
    # Convert dates if needed
    if df["start_date_local"].dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("start_date_local").str.to_datetime()
        )
    
    # Convert numeric columns from string if needed
    # Identify available numeric columns
    numeric_cols = ["moving_time", "distance", "elevation_gain", "total_ascent"]
    # Add Intervals columns if present
    if "icu_fitness" in df.columns:
        numeric_cols.append("icu_fitness")
    if "icu_fatigue" in df.columns:
        numeric_cols.append("icu_fatigue")
    if "icu_training_load" in df.columns:
        numeric_cols.append("icu_training_load")
    # Add Garmin columns if present
    if "training_stress_score" in df.columns:
        numeric_cols.append("training_stress_score")
    for col in numeric_cols:
        if col in df.columns and df[col].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col(col).cast(pl.Float64, strict=False)
            )
    
    # Group by date and aggregate - only include available columns
    agg_exprs = []
    
    if "icu_fitness" in df.columns:
        agg_exprs.append(pl.col("icu_fitness").mean())
    if "icu_fatigue" in df.columns:
        agg_exprs.append(pl.col("icu_fatigue").mean())
    # Handle both Intervals and Garmin training load for aggregation
    if "icu_training_load" in df.columns:
        agg_exprs.append(pl.col("icu_training_load").sum())
    elif "training_stress_score" in df.columns:
        agg_exprs.append(pl.col("training_stress_score").sum())
    if "moving_time" in df.columns:
        agg_exprs.append(pl.col("moving_time").sum().alias("total_time"))
    if "distance" in df.columns:
        agg_exprs.append(pl.col("distance").sum().alias("total_distance"))
    
    if not agg_exprs:
        # No metrics to aggregate
        return pl.DataFrame({"date": []})
    
    daily = df.group_by(
        pl.col("start_date_local").dt.date().alias("date")
    ).agg(agg_exprs).sort("date")
    
    return daily


def prepare_calendar_data(df: pl.DataFrame) -> pl.DataFrame:
    """Prepare data for calendar heatmap."""
    if df.is_empty():
        return pl.DataFrame()
    
    # Convert dates if needed
    if df["start_date_local"].dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("start_date_local").str.to_datetime()
        )
    
    # Filter out null dates before dt operations
    df = df.filter(pl.col("start_date_local").is_not_null())
    if df.is_empty():
        return pl.DataFrame()

    # Add week and weekday columns
    calendar_df = df.with_columns([
        pl.col("start_date_local").dt.week().alias("week"),
        pl.col("start_date_local").dt.weekday().alias("weekday"),
    ])
    
    # Group by week and weekday - only include available columns
    agg_exprs = [pl.len().alias("activity_count")]
    
    # Handle both Intervals and Garmin training load
    if "icu_training_load" in calendar_df.columns:
        agg_exprs.append(pl.col("icu_training_load").sum().alias("training_load"))
    elif "training_stress_score" in calendar_df.columns:
        agg_exprs.append(pl.col("training_stress_score").sum().alias("training_load"))
    else:
        # Fallback to duration-based estimate
        agg_exprs.append((pl.col("moving_time").sum() / 60).alias("training_load"))
    
    calendar_agg = calendar_df.group_by(["week", "weekday"]).agg(agg_exprs)
    
    return calendar_agg


def add_fit_analysis_section(df: pl.DataFrame):
    """Add comprehensive FIT data analysis section to dashboard."""
    # Check if we have FIT data
    fit_columns = [col for col in df.columns if col.startswith('fit_')]
    
    if not fit_columns:
        return  # No FIT data available
    
    # Count activities with FIT data
    fit_activities = df.filter(pl.col("has_fit_analysis") == True) if "has_fit_analysis" in df.columns else pl.DataFrame()
    
    if fit_activities.is_empty():
        st.header("🎯 FIT File Analysis")
        st.info("💡 No activities with FIT file analysis found. Run 'python scripts/analyze_existing_fit.py' to analyze existing FIT files.")
        return
    
    st.header("🎯 FIT File Analysis")
    st.caption(f"Detailed second-by-second telemetry from {len(fit_activities)}/{len(df)} activities")
    
    # FIT Data Overview
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("FIT Activities", len(fit_activities))
    
    with col2:
        if "fit_data_points" in fit_activities.columns:
            total_points = fit_activities["fit_data_points"].sum()
            st.metric("Data Points", f"{total_points:,}")
    
    with col3:
        if "fit_power_normalized" in fit_activities.columns:
            avg_np = fit_activities["fit_power_normalized"].mean()
            st.metric("Avg NP", f"{avg_np:.0f}W" if avg_np else "N/A")
    
    with col4:
        if "fit_hr_avg" in fit_activities.columns:
            avg_hr = fit_activities["fit_hr_avg"].mean()
            st.metric("Avg HR", f"{avg_hr:.0f}" if avg_hr else "N/A")
    
    with col5:
        if "fit_cadence_avg" in fit_activities.columns:
            avg_cadence = fit_activities["fit_cadence_avg"].mean()
            st.metric("Avg Cadence", f"{avg_cadence:.0f}" if avg_cadence else "N/A")
    
    # Detailed FIT Analysis Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔋 Power Analysis", "❤️ Heart Rate", "🔄 Zones", "⚙️ Pedaling Dynamics", "🌡️ Environmental"
    ])
    
    with tab1:
        add_power_analysis_tab(fit_activities)
    
    with tab2:
        add_heart_rate_tab(fit_activities)
    
    with tab3:
        add_zones_tab(fit_activities)
    
    with tab4:
        add_pedaling_dynamics_tab(fit_activities)
    
    with tab5:
        add_environmental_tab(fit_activities)


def add_power_analysis_tab(df: pl.DataFrame):
    """Add detailed power analysis tab."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Power Metrics Comparison")
        
        # Power metrics comparison
        power_metrics = []
        labels = []
        
        if "fit_power_avg" in df.columns:
            avg_power = df["fit_power_avg"].mean()
            if avg_power:
                power_metrics.append(avg_power)
                labels.append("Average Power")
        
        if "fit_power_normalized" in df.columns:
            np_power = df["fit_power_normalized"].mean()
            if np_power:
                power_metrics.append(np_power)
                labels.append("Normalized Power")
        
        if "fit_power_max" in df.columns:
            max_power = df["fit_power_max"].mean()
            if max_power:
                power_metrics.append(max_power)
                labels.append("Max Power (Avg)")
        
        if power_metrics:
            fig = go.Figure(data=[
                go.Bar(x=labels, y=power_metrics, marker_color=["lightblue", "orange", "red"])
            ])
            
            fig.update_layout(
                title="Power Metrics (Average Across Activities)",
                yaxis_title="Power (W)",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Variability Index")
        
        if "fit_power_variability_index" in df.columns:
            vi_data = df.filter(pl.col("fit_power_variability_index").is_not_null())["fit_power_variability_index"]
            
            if len(vi_data) > 0:
                # VI distribution
                fig = go.Figure(data=[go.Histogram(
                    x=vi_data.to_list(),
                    nbinsx=20,
                    marker_color="green",
                    opacity=0.7
                )])
                
                fig.add_vline(x=1.05, line_dash="dash", line_color="red", 
                             annotation_text="Ideal VI < 1.05")
                
                fig.update_layout(
                    title="Power Variability Index Distribution",
                    xaxis_title="Variability Index",
                    yaxis_title="Frequency",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # VI stats
                avg_vi = vi_data.mean()
                st.metric("Average VI", f"{avg_vi:.3f}", 
                         delta=f"{'Good' if avg_vi < 1.05 else 'High'}")
    
    # Power over time
    if "start_date_local" in df.columns and "fit_power_normalized" in df.columns:
        st.subheader("Power Progression")
        
        power_df = df.filter(pl.col("fit_power_normalized").is_not_null())
        
        if not power_df.is_empty():
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=power_df["start_date_local"],
                y=power_df["fit_power_normalized"],
                mode="lines+markers",
                name="Normalized Power",
                line=dict(color="orange", width=2),
                marker=dict(size=8)
            ))
            
            if "fit_power_avg" in power_df.columns:
                fig.add_trace(go.Scatter(
                    x=power_df["start_date_local"],
                    y=power_df["fit_power_avg"],
                    mode="lines+markers",
                    name="Average Power",
                    line=dict(color="lightblue", width=2),
                    marker=dict(size=6)
                ))
            
            fig.update_layout(
                title="Power Over Time",
                xaxis_title="Date",
                yaxis_title="Power (W)",
                height=300,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)


def add_heart_rate_tab(df: pl.DataFrame):
    """Add heart rate analysis tab."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Heart Rate Distribution")
        
        if "fit_hr_avg" in df.columns and "fit_hr_max" in df.columns:
            hr_avg_data = df.filter(pl.col("fit_hr_avg").is_not_null())["fit_hr_avg"]
            hr_max_data = df.filter(pl.col("fit_hr_max").is_not_null())["fit_hr_max"]
            
            if len(hr_avg_data) > 0 and len(hr_max_data) > 0:
                fig = go.Figure()
                
                fig.add_trace(go.Box(
                    y=hr_avg_data.to_list(),
                    name="Average HR",
                    marker_color="lightblue"
                ))
                
                fig.add_trace(go.Box(
                    y=hr_max_data.to_list(),
                    name="Max HR",
                    marker_color="red"
                ))
                
                fig.update_layout(
                    title="Heart Rate Distribution",
                    yaxis_title="Heart Rate (BPM)",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("HR Variability")
        
        if "fit_hr_std" in df.columns:
            hr_std_data = df.filter(pl.col("fit_hr_std").is_not_null())["fit_hr_std"]
            
            if len(hr_std_data) > 0:
                fig = go.Figure(data=[go.Histogram(
                    x=hr_std_data.to_list(),
                    nbinsx=15,
                    marker_color="purple",
                    opacity=0.7
                )])
                
                fig.update_layout(
                    title="Heart Rate Variability (Std Dev)",
                    xaxis_title="HR Standard Deviation",
                    yaxis_title="Frequency",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                avg_hrv = hr_std_data.mean()
                st.metric("Average HR Variability", f"{avg_hrv:.1f} BPM")
    
    # HR efficiency (Power per HR beat)
    if "fit_power_normalized" in df.columns and "fit_hr_avg" in df.columns:
        st.subheader("Heart Rate Efficiency")
        
        efficiency_df = df.filter(
            (pl.col("fit_power_normalized").is_not_null()) &
            (pl.col("fit_hr_avg").is_not_null()) &
            (pl.col("fit_hr_avg") > 0)
        ).with_columns(
            (pl.col("fit_power_normalized") / pl.col("fit_hr_avg")).alias("power_per_beat")
        )
        
        if not efficiency_df.is_empty():
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=efficiency_df["start_date_local"],
                y=efficiency_df["power_per_beat"],
                mode="lines+markers",
                name="Power per Beat",
                line=dict(color="green", width=2),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Heart Rate Efficiency Over Time (Power per BPM)",
                xaxis_title="Date",
                yaxis_title="Watts per BPM",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)


def add_zones_tab(df: pl.DataFrame):
    """Add training zones analysis tab."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Heart Rate Zones")
        
        # HR zones distribution
        hr_zone_cols = [f"fit_hr_zones_zone{i}" for i in range(1, 6)]
        hr_zones_data = {}
        
        for i, col in enumerate(hr_zone_cols, 1):
            if col in df.columns:
                zone_data = df.filter(pl.col(col).is_not_null())[col]
                if len(zone_data) > 0:
                    hr_zones_data[f"Zone {i}"] = zone_data.mean()
        
        if hr_zones_data:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(hr_zones_data.keys()),
                    y=list(hr_zones_data.values()),
                    marker_color=px.colors.sequential.Reds[2:7]
                )
            ])
            
            fig.update_layout(
                title="Average Time in HR Zones (%)",
                yaxis_title="Percentage",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Power Zones")
        
        # Power zones distribution
        power_zone_cols = [f"fit_power_zones_zone{i}" for i in range(1, 8)]
        power_zones_data = {}
        
        for i, col in enumerate(power_zone_cols, 1):
            if col in df.columns:
                zone_data = df.filter(pl.col(col).is_not_null())[col]
                if len(zone_data) > 0:
                    power_zones_data[f"Z{i}"] = zone_data.mean()
        
        if power_zones_data:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(power_zones_data.keys()),
                    y=list(power_zones_data.values()),
                    marker_color=px.colors.sequential.Viridis
                )
            ])
            
            fig.update_layout(
                title="Average Time in Power Zones (%)",
                yaxis_title="Percentage",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Zone balance analysis
    st.subheader("Training Balance Analysis")
    
    if power_zones_data:
        # Polarized training analysis
        z1_z2 = power_zones_data.get("Z1", 0) + power_zones_data.get("Z2", 0)  # Easy
        z3_z4 = power_zones_data.get("Z3", 0) + power_zones_data.get("Z4", 0)  # Moderate
        z5_z6_z7 = power_zones_data.get("Z5", 0) + power_zones_data.get("Z6", 0) + power_zones_data.get("Z7", 0)  # Hard
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Easy (Z1-Z2)", f"{z1_z2:.1f}%", 
                     delta="Target: 80%" if z1_z2 >= 75 else "Target: 80%")
        
        with col2:
            st.metric("Moderate (Z3-Z4)", f"{z3_z4:.1f}%",
                     delta="Target: 5-10%")
        
        with col3:
            st.metric("Hard (Z5-Z7)", f"{z5_z6_z7:.1f}%",
                     delta="Target: 10-15%")
        
        # Polarization index
        if z3_z4 > 0:
            polarization = (z1_z2 + z5_z6_z7) / z3_z4
            st.metric("Polarization Index", f"{polarization:.2f}",
                     delta="Higher = More Polarized")


def add_pedaling_dynamics_tab(df: pl.DataFrame):
    """Add pedaling dynamics analysis tab."""
    st.subheader("Pedaling Balance & Efficiency")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Left-right balance
        if "fit_left_right_balance" in df.columns:
            st.subheader("Pedaling Balance")
            
            balance_data = df.filter(pl.col("fit_left_right_balance").is_not_null())["fit_left_right_balance"]
            
            if len(balance_data) > 0:
                # Convert string values if needed
                balance_values = []
                for val in balance_data:
                    if isinstance(val, (int, float)):
                        balance_values.append(val)
                    elif isinstance(val, str):
                        try:
                            balance_values.append(float(val))
                        except:
                            balance_values.append(50.0)  # Default balanced
                
                if balance_values:
                    fig = go.Figure(data=[go.Histogram(
                        x=balance_values,
                        nbinsx=20,
                        marker_color="lightgreen",
                        opacity=0.7
                    )])
                    
                    fig.add_vline(x=50, line_dash="dash", line_color="red", 
                                 annotation_text="Perfect Balance")
                    
                    fig.update_layout(
                        title="Left-Right Pedaling Balance Distribution",
                        xaxis_title="Balance (%)",
                        yaxis_title="Frequency",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    avg_balance = sum(balance_values) / len(balance_values)
                    imbalance = abs(50 - avg_balance)
                    st.metric("Average Balance", f"{avg_balance:.1f}%",
                             delta=f"Imbalance: {imbalance:.1f}%")
    
    with col2:
        st.subheader("Cadence Consistency")
        
        if "fit_cadence_avg" in df.columns:
            cadence_data = df.filter(pl.col("fit_cadence_avg").is_not_null())["fit_cadence_avg"]
            
            if len(cadence_data) > 0:
                fig = go.Figure(data=[go.Box(
                    y=cadence_data.to_list(),
                    name="Cadence",
                    marker_color="orange"
                )])
                
                fig.update_layout(
                    title="Cadence Distribution",
                    yaxis_title="Cadence (RPM)",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                avg_cadence = cadence_data.mean()
                std_cadence = cadence_data.std()
                st.metric("Average Cadence", f"{avg_cadence:.0f} RPM",
                         delta=f"Std: {std_cadence:.1f}")
    
    # Torque effectiveness if available
    torque_cols = ["fit_avg_left_torque_effectiveness", "fit_avg_right_torque_effectiveness"]
    torque_data = {}
    
    for col in torque_cols:
        if col in df.columns:
            data = df.filter(pl.col(col).is_not_null())[col]
            if len(data) > 0:
                side = "Left" if "left" in col else "Right"
                torque_data[side] = data.mean()
    
    if torque_data:
        st.subheader("Torque Effectiveness")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "Left" in torque_data:
                st.metric("Left TE", f"{torque_data['Left']:.1f}%")
        
        with col2:
            if "Right" in torque_data:
                st.metric("Right TE", f"{torque_data['Right']:.1f}%")
        
        with col3:
            if len(torque_data) == 2:
                avg_te = sum(torque_data.values()) / len(torque_data)
                st.metric("Average TE", f"{avg_te:.1f}%")


def add_environmental_tab(df: pl.DataFrame):
    """Add environmental conditions analysis tab."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Temperature Analysis")
        
        if "fit_temp_avg" in df.columns:
            temp_data = df.filter(pl.col("fit_temp_avg").is_not_null())["fit_temp_avg"]
            
            if len(temp_data) > 0:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df.filter(pl.col("fit_temp_avg").is_not_null())["start_date_local"],
                    y=temp_data,
                    mode="lines+markers",
                    name="Temperature",
                    line=dict(color="red", width=2),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title="Temperature During Rides",
                    xaxis_title="Date",
                    yaxis_title="Temperature (°C)",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                avg_temp = temp_data.mean()
                min_temp = temp_data.min()
                max_temp = temp_data.max()
                
                col1_1, col1_2, col1_3 = st.columns(3)
                with col1_1:
                    st.metric("Avg Temp", f"{avg_temp:.1f}°C")
                with col1_2:
                    st.metric("Min Temp", f"{min_temp:.1f}°C")
                with col1_3:
                    st.metric("Max Temp", f"{max_temp:.1f}°C")
    
    with col2:
        st.subheader("Altitude Gain Analysis")
        
        if "fit_altitude_gain" in df.columns:
            alt_data = df.filter(pl.col("fit_altitude_gain").is_not_null())["fit_altitude_gain"]
            
            if len(alt_data) > 0:
                fig = go.Figure(data=[go.Histogram(
                    x=alt_data.to_list(),
                    nbinsx=20,
                    marker_color="brown",
                    opacity=0.7
                )])
                
                fig.update_layout(
                    title="Altitude Gain Distribution",
                    xaxis_title="Elevation Gain (m)",
                    yaxis_title="Frequency",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                avg_alt = alt_data.mean()
                total_alt = alt_data.sum()
                st.metric("Avg Elevation/Ride", f"{avg_alt:.0f}m")
                st.metric("Total Elevation", f"{total_alt:,.0f}m")
    
    # Data Quality Section
    st.subheader("Data Quality & Coverage")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "fit_data_points" in df.columns:
            points_data = df.filter(pl.col("fit_data_points").is_not_null())["fit_data_points"]
            if len(points_data) > 0:
                avg_points = points_data.mean()
                st.metric("Avg Data Points/Ride", f"{avg_points:,.0f}")
    
    with col2:
        if "fit_lap_count" in df.columns:
            laps_data = df.filter(pl.col("fit_lap_count").is_not_null())["fit_lap_count"]
            if len(laps_data) > 0:
                avg_laps = laps_data.mean()
                st.metric("Avg Laps/Ride", f"{avg_laps:.1f}")
    
    with col3:
        # Calculate data density (points per minute)
        if "fit_data_points" in df.columns and "moving_time" in df.columns:
            density_df = df.filter(
                (pl.col("fit_data_points").is_not_null()) &
                (pl.col("moving_time").is_not_null()) &
                (pl.col("moving_time") > 0)
            ).with_columns(
                (pl.col("fit_data_points") / (pl.col("moving_time") / 60)).alias("data_density")
            )
            
            if not density_df.is_empty():
                avg_density = density_df["data_density"].mean()
                st.metric("Data Density", f"{avg_density:.1f} pts/min")


if __name__ == "__main__":
    main()