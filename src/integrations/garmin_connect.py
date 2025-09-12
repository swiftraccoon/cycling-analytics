"""Garmin Connect integration for fetching detailed activity data with FIT files."""

import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
import zipfile
from io import BytesIO

import polars as pl
from garth.exc import GarthHTTPError

# Add python-garminconnect to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python-garminconnect"))

from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)

logger = logging.getLogger(__name__)


class GarminConnectClient:
    """Client for interacting with Garmin Connect API."""
    
    def __init__(
        self,
        email: Optional[str] = None,
        password: Optional[str] = None,
        tokenstore: Optional[str] = None
    ):
        """Initialize Garmin Connect client.
        
        Args:
            email: Garmin Connect email
            password: Garmin Connect password
            tokenstore: Path to store OAuth tokens for reuse
        """
        self.email = email or os.environ.get("GARMIN_EMAIL")
        self.password = password or os.environ.get("GARMIN_PASSWORD")
        self.tokenstore = tokenstore or os.path.expanduser("~/.garminconnect")
        self.api = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Garmin Connect."""
        try:
            # Try to login using saved tokens
            logger.info(f"Attempting to login using tokens from {self.tokenstore}")
            self.api = Garmin()
            self.api.login(self.tokenstore)
            logger.info("Successfully logged in using saved tokens")
            
        except (FileNotFoundError, GarthHTTPError, GarminConnectAuthenticationError):
            # Need to login with credentials
            if not self.email or not self.password:
                raise ValueError(
                    "Garmin Connect credentials not provided. "
                    "Set GARMIN_EMAIL and GARMIN_PASSWORD environment variables "
                    "or pass them to the constructor."
                )
            
            logger.info("Logging in with email/password")
            try:
                self.api = Garmin(
                    email=self.email,
                    password=self.password,
                    is_cn=False,
                    return_on_mfa=True
                )
                
                result1, result2 = self.api.login()
                
                if result1 == "needs_mfa":
                    # Handle MFA if needed
                    mfa_code = input("Enter Garmin Connect MFA code: ")
                    self.api.resume_login(result2, mfa_code)
                
                # Save tokens for future use
                self.api.garth.dump(self.tokenstore)
                logger.info(f"OAuth tokens saved to {self.tokenstore}")
                
                # Re-login with tokens
                self.api.login(self.tokenstore)
                
            except Exception as e:
                logger.error(f"Failed to login to Garmin Connect: {e}")
                raise
    
    def get_activities(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        activity_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Fetch activities from Garmin Connect.
        
        Args:
            start_date: Start date for activities (default: 30 days ago)
            end_date: End date for activities (default: today)
            activity_type: Filter by activity type (e.g., 'cycling', 'running')
            limit: Maximum number of activities to fetch
            
        Returns:
            List of activity dictionaries
        """
        if not self.api:
            raise RuntimeError("Not connected to Garmin Connect")
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        logger.info(f"Fetching activities from {start_str} to {end_str} (limit: {limit})")
        
        try:
            # Note: get_activities_by_date doesn't support limit, so we fetch all and slice
            # Or use get_activities with start/limit but it doesn't support date filtering
            # For now, we'll fetch by date and then limit the results
            activities = self.api.get_activities_by_date(
                start_str,
                end_str,
                activity_type
            )
            
            # Apply limit manually
            if len(activities) > limit:
                logger.info(f"Got {len(activities)} activities, limiting to {limit}")
                activities = activities[:limit]
            
            logger.info(f"Fetched {len(activities)} activities")
            return activities
            
        except GarminConnectTooManyRequestsError:
            logger.error("Too many requests to Garmin Connect. Please try again later.")
            raise
        except Exception as e:
            logger.error(f"Error fetching activities: {e}")
            raise
    
    def download_activity_fit(self, activity_id: str) -> bytes:
        """Download FIT file for an activity.
        
        Args:
            activity_id: Activity ID
            
        Returns:
            FIT file data as bytes
        """
        if not self.api:
            raise RuntimeError("Not connected to Garmin Connect")
        
        logger.debug(f"Downloading FIT file for activity {activity_id}")
        
        try:
            # Download original format (usually FIT in a ZIP)
            zip_data = self.api.download_activity(
                activity_id,
                dl_fmt=self.api.ActivityDownloadFormat.ORIGINAL
            )
            
            # Extract FIT file from ZIP
            with zipfile.ZipFile(BytesIO(zip_data)) as zip_file:
                for filename in zip_file.namelist():
                    if filename.endswith('.fit'):
                        return zip_file.read(filename)
            
            logger.warning(f"No FIT file found in download for activity {activity_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error downloading FIT file for activity {activity_id}: {e}")
            return None
    
    def get_activity_details(self, activity_id: str) -> Dict:
        """Get detailed information for an activity.
        
        Args:
            activity_id: Activity ID
            
        Returns:
            Activity details dictionary
        """
        if not self.api:
            raise RuntimeError("Not connected to Garmin Connect")
        
        try:
            details = self.api.get_activity(activity_id)
            return details
        except Exception as e:
            logger.error(f"Error fetching activity details for {activity_id}: {e}")
            return None
    
    def get_activity_splits(self, activity_id: str) -> List[Dict]:
        """Get splits/laps for an activity.
        
        Args:
            activity_id: Activity ID
            
        Returns:
            List of split dictionaries
        """
        if not self.api:
            raise RuntimeError("Not connected to Garmin Connect")
        
        try:
            splits = self.api.get_activity_splits(activity_id)
            return splits
        except Exception as e:
            logger.error(f"Error fetching splits for activity {activity_id}: {e}")
            return []
    
    def get_activity_hr_timezones(self, activity_id: str) -> List[Dict]:
        """Get heart rate time in zones for an activity.
        
        Args:
            activity_id: Activity ID
            
        Returns:
            Heart rate zones data
        """
        if not self.api:
            raise RuntimeError("Not connected to Garmin Connect")
        
        try:
            hr_zones = self.api.get_activity_hr_in_timezones(activity_id)
            return hr_zones
        except Exception as e:
            logger.error(f"Error fetching HR zones for activity {activity_id}: {e}")
            return []
    
    def get_user_profile(self) -> Dict:
        """Get user profile information."""
        if not self.api:
            raise RuntimeError("Not connected to Garmin Connect")
        
        try:
            return self.api.get_full_name()
        except Exception as e:
            logger.error(f"Error fetching user profile: {e}")
            return {}
    
    def get_stats(self, date: Optional[datetime] = None) -> Dict:
        """Get daily stats for a given date.
        
        Args:
            date: Date to get stats for (default: today)
            
        Returns:
            Stats dictionary
        """
        if not self.api:
            raise RuntimeError("Not connected to Garmin Connect")
        
        if not date:
            date = datetime.now()
        
        date_str = date.strftime("%Y-%m-%d")
        
        try:
            return self.api.get_stats(date_str)
        except Exception as e:
            logger.error(f"Error fetching stats for {date_str}: {e}")
            return {}


class GarminConnectSyncManager:
    """Manages synchronization of Garmin Connect data."""
    
    def __init__(
        self,
        client: GarminConnectClient,
        state_file: Optional[Path] = None,
        fit_storage_dir: Optional[Path] = None
    ):
        """Initialize sync manager.
        
        Args:
            client: GarminConnectClient instance
            state_file: Path to store sync state
            fit_storage_dir: Directory to store downloaded FIT files
        """
        self.client = client
        self.state_file = state_file or Path("data/garmin_sync_state.json")
        self.fit_storage_dir = fit_storage_dir or Path("data/bronze/fit_files")
        self.fit_storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.sync_state = self._load_sync_state()
    
    def _load_sync_state(self) -> Dict:
        """Load sync state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading sync state: {e}")
        
        return {
            "last_sync_date": None,
            "last_sync_count": 0,
            "synced_activity_ids": []
        }
    
    def _save_sync_state(self):
        """Save sync state to file."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(self.sync_state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving sync state: {e}")
    
    def sync_activities(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        download_fit: bool = True,
        force_full: bool = False,
        limit: int = 100
    ) -> pl.DataFrame:
        """Sync activities from Garmin Connect.
        
        Args:
            start_date: Start date for sync
            end_date: End date for sync
            download_fit: Whether to download FIT files
            force_full: Force full sync instead of incremental
            limit: Maximum number of activities to sync
            
        Returns:
            DataFrame with synced activities
        """
        # Determine sync range
        if start_date:
            # User provided explicit start date - use it
            logger.info("Using user-provided date range")
        elif force_full or not self.sync_state.get("last_sync_date"):
            # Full sync - last 90 days by default
            start_date = datetime.now() - timedelta(days=90)
            logger.info("Performing full sync")
        else:
            # Incremental sync since last sync
            last_sync = datetime.fromisoformat(self.sync_state["last_sync_date"])
            start_date = last_sync - timedelta(days=1)  # Overlap by 1 day
            logger.info(f"Performing incremental sync since {start_date}")
        
        if not end_date:
            end_date = datetime.now()
        
        # Fetch activities
        activities = self.client.get_activities(
            start_date=start_date,
            end_date=end_date,
            activity_type="cycling",  # Focus on cycling
            limit=limit
        )
        
        if not activities:
            logger.info("No new activities to sync")
            return pl.DataFrame()
        
        # Process activities
        processed_activities = []
        
        for activity in activities:
            activity_id = str(activity.get("activityId"))
            
            # Skip if already synced (unless force_full)
            if not force_full and activity_id in self.sync_state.get("synced_activity_ids", []):
                logger.debug(f"Skipping already synced activity {activity_id}")
                continue
            
            # Build activity record
            activity_data = self._process_activity(activity)
            
            # Download FIT file if requested
            if download_fit:
                fit_path = self._download_and_store_fit(activity_id, activity_data.get("name", "Unknown"))
                if fit_path:
                    activity_data["fit_file_path"] = str(fit_path)
            
            # Get additional details
            details = self.client.get_activity_details(activity_id)
            if details:
                activity_data.update(self._extract_additional_metrics(details))
            
            # Get splits data
            splits = self.client.get_activity_splits(activity_id)
            if splits:
                activity_data["splits_data"] = json.dumps(splits)
            
            # Get HR zones
            hr_zones = self.client.get_activity_hr_timezones(activity_id)
            if hr_zones:
                activity_data["hr_zones_data"] = json.dumps(hr_zones)
            
            processed_activities.append(activity_data)
            
            # Update sync state
            if activity_id not in self.sync_state.get("synced_activity_ids", []):
                self.sync_state["synced_activity_ids"].append(activity_id)
        
        # Create DataFrame
        if processed_activities:
            df = pl.DataFrame(processed_activities)
            
            # Update sync state
            self.sync_state["last_sync_date"] = datetime.now().isoformat()
            self.sync_state["last_sync_count"] = len(processed_activities)
            self._save_sync_state()
            
            logger.info(f"Synced {len(processed_activities)} activities from Garmin Connect")
            return df
        
        return pl.DataFrame()
    
    def _process_activity(self, activity: Dict) -> Dict:
        """Process raw activity data from Garmin Connect.
        
        Args:
            activity: Raw activity dictionary from API
            
        Returns:
            Processed activity dictionary
        """
        # Map Garmin fields to our schema
        processed = {
            "id": f"garmin_{activity.get('activityId')}",
            "garmin_activity_id": str(activity.get("activityId")),
            "name": activity.get("activityName", ""),
            "type": activity.get("activityType", {}).get("typeKey", ""),
            "sport_type": activity.get("sportTypeDTO", {}).get("sportTypeKey", ""),
            "start_date_local": self._parse_datetime(activity.get("startTimeLocal")),
            "start_date_gmt": self._parse_datetime(activity.get("startTimeGMT")),
            "timezone": activity.get("timeZoneDTO", {}).get("displayName", ""),
            "distance": activity.get("distance", 0),  # meters
            "duration": activity.get("duration", 0),  # seconds
            "elapsed_duration": activity.get("elapsedDuration", 0),
            "moving_duration": activity.get("movingDuration", 0),
            "average_speed": activity.get("averageSpeed", 0),  # m/s
            "max_speed": activity.get("maxSpeed", 0),
            "elevation_gain": activity.get("elevationGain", 0),
            "elevation_loss": activity.get("elevationLoss", 0),
            "average_hr": activity.get("averageHR"),
            "max_hr": activity.get("maxHR"),
            "calories": activity.get("calories"),
            "bmr_calories": activity.get("bmrCalories"),
            "average_cadence": activity.get("averageRunningCadenceInStepsPerMinute"),
            "max_cadence": activity.get("maxRunningCadenceInStepsPerMinute"),
            "average_power": activity.get("avgPower"),
            "max_power": activity.get("maxPower"),
            "normalized_power": activity.get("normPower"),
            "training_stress_score": activity.get("trainingStressScore"),
            "intensity_factor": activity.get("intensityFactor"),
            "avg_stride_length": activity.get("avgStrideLength"),
            "vo2_max": activity.get("vO2MaxValue"),
            "lactate_threshold_hr": activity.get("lactateThresholdBpm"),
            "device_name": activity.get("deviceDTO", {}).get("displayName"),
            "max_temperature": activity.get("maxTemp"),
            "min_temperature": activity.get("minTemp"),
            "start_latitude": activity.get("startLatitude"),
            "start_longitude": activity.get("startLongitude"),
            "end_latitude": activity.get("endLatitude"),
            "end_longitude": activity.get("endLongitude"),
            "source": "garmin_connect",
            "has_polyline": activity.get("hasPolyline", False),
            "has_splits": activity.get("hasSplits", False),
            "pr_count": activity.get("pr", 0),
        }
        
        return processed
    
    def _extract_additional_metrics(self, details: Dict) -> Dict:
        """Extract additional metrics from detailed activity data.
        
        Args:
            details: Detailed activity data
            
        Returns:
            Dictionary of additional metrics
        """
        metrics = {}
        
        # Performance metrics
        if "summaryDTO" in details:
            summary = details["summaryDTO"]
            metrics["avg_ground_contact_time"] = summary.get("avgGroundContactTime")
            metrics["avg_vertical_oscillation"] = summary.get("avgVerticalOscillation")
            metrics["avg_ground_contact_balance"] = summary.get("avgGroundContactBalance")
            metrics["avg_vertical_ratio"] = summary.get("avgVerticalRatio")
            metrics["training_effect"] = summary.get("trainingEffect")
            metrics["anaerobic_training_effect"] = summary.get("anaerobicTrainingEffect")
        
        # Power metrics
        if "powerDTO" in details:
            power = details["powerDTO"]
            metrics["avg_left_balance"] = power.get("avgLeftBalance")
            metrics["avg_right_balance"] = power.get("avgRightBalance")
            metrics["functional_threshold_power"] = power.get("functionalThresholdPower")
        
        # Weather data
        if "weatherDTO" in details:
            weather = details["weatherDTO"]
            metrics["weather_temp"] = weather.get("temp")
            metrics["weather_humidity"] = weather.get("relativeHumidity")
            metrics["weather_wind_speed"] = weather.get("windSpeed")
            metrics["weather_wind_direction"] = weather.get("windDirection")
        
        return metrics
    
    def _download_and_store_fit(self, activity_id: str, activity_name: str) -> Optional[Path]:
        """Download and store FIT file for an activity.
        
        Args:
            activity_id: Activity ID
            activity_name: Activity name for filename
            
        Returns:
            Path to stored FIT file or None if failed
        """
        try:
            fit_data = self.client.download_activity_fit(activity_id)
            
            if fit_data:
                # Create filename
                safe_name = "".join(c for c in activity_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_name = safe_name[:50]  # Limit length
                filename = f"{activity_id}_{safe_name}.fit"
                fit_path = self.fit_storage_dir / filename
                
                # Save FIT file
                with open(fit_path, 'wb') as f:
                    f.write(fit_data)
                
                logger.debug(f"Saved FIT file to {fit_path}")
                return fit_path
            
        except Exception as e:
            logger.error(f"Error downloading FIT file for activity {activity_id}: {e}")
        
        return None
    
    def _parse_datetime(self, datetime_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string from Garmin Connect.
        
        Args:
            datetime_str: Datetime string
            
        Returns:
            Parsed datetime or None
        """
        if not datetime_str:
            return None
        
        try:
            # Garmin uses milliseconds since epoch
            if isinstance(datetime_str, (int, float)):
                return datetime.fromtimestamp(datetime_str / 1000, tz=timezone.utc)
            
            # Try parsing ISO format
            return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        except Exception as e:
            logger.debug(f"Could not parse datetime: {datetime_str} - {e}")
            return None
    
    def get_sync_status(self) -> Dict:
        """Get current sync status.
        
        Returns:
            Sync status dictionary
        """
        return {
            "last_sync_date": self.sync_state.get("last_sync_date"),
            "last_sync_count": self.sync_state.get("last_sync_count", 0),
            "total_synced_activities": len(self.sync_state.get("synced_activity_ids", [])),
            "fit_files_directory": str(self.fit_storage_dir),
            "fit_files_count": len(list(self.fit_storage_dir.glob("*.fit")))
        }