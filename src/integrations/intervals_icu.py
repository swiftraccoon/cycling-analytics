"""Intervals.icu API integration for fetching cycling activities.

This module provides efficient API access to Intervals.icu with:
- Incremental sync from last import date
- Rate limiting and retry logic
- Minimal API calls
- Data integrity preservation
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import time
import requests
import polars as pl
from pathlib import Path

logger = logging.getLogger(__name__)


class IntervalsICUClient:
    """Client for Intervals.icu API with efficient data fetching."""
    
    BASE_URL = "https://intervals.icu/api/v1"
    
    def __init__(self, athlete_id: str, api_key: str):
        """Initialize Intervals.icu client.
        
        Args:
            athlete_id: The athlete's ID (can be username or numeric ID)
            api_key: API key from Settings -> Developer Settings
        """
        self.athlete_id = athlete_id
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}"
        })
        
        # Rate limiting: Intervals.icu allows 100 requests per minute
        self.rate_limit_calls = 0
        self.rate_limit_reset = time.time()
        self.max_calls_per_minute = 90  # Conservative limit
        
    def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Reset counter if minute has passed
        if current_time - self.rate_limit_reset >= 60:
            self.rate_limit_calls = 0
            self.rate_limit_reset = current_time
        
        # If approaching limit, wait
        if self.rate_limit_calls >= self.max_calls_per_minute:
            wait_time = 60 - (current_time - self.rate_limit_reset)
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
                self.rate_limit_calls = 0
                self.rate_limit_reset = time.time()
        
        self.rate_limit_calls += 1
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Any]:
        """Make an API request with error handling and retries.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional request parameters
            
        Returns:
            Response data or None if error
        """
        self._check_rate_limit()
        
        url = f"{self.BASE_URL}{endpoint}"
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.session.request(method, url, **kwargs)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Too Many Requests
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited, waiting {retry_after} seconds")
                    time.sleep(retry_after)
                elif response.status_code == 401:
                    logger.error("Authentication failed. Check API key.")
                    return None
                elif response.status_code == 404:
                    logger.warning(f"Resource not found: {endpoint}")
                    return None
                else:
                    logger.warning(f"Request failed with status {response.status_code}")
                    
            except requests.RequestException as e:
                logger.error(f"Request error: {e}")
                
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        
        return None
    
    def get_activities(
        self,
        oldest: Optional[datetime] = None,
        newest: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Fetch activities for a date range.
        
        Args:
            oldest: Start date (ISO-8601 format)
            newest: End date (defaults to now)
            limit: Maximum number of activities to return
            
        Returns:
            List of activity dictionaries
        """
        params = {}
        
        if oldest:
            params["oldest"] = oldest.strftime("%Y-%m-%d")
        else:
            # Default to 30 days ago if not specified
            params["oldest"] = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        if newest:
            params["newest"] = newest.strftime("%Y-%m-%d")
        
        if limit:
            params["limit"] = limit
        
        endpoint = f"/athlete/{self.athlete_id}/activities"
        
        logger.info(f"Fetching activities from {params.get('oldest')} to {params.get('newest', 'now')}")
        
        activities = self._make_request("GET", endpoint, params=params)
        
        if activities is None:
            logger.error("Failed to fetch activities")
            return []
        
        # Filter out Strava stub activities (they have minimal data)
        full_activities = []
        for activity in activities:
            # Strava activities typically have very few fields
            if activity.get("icu_sync_error") != "Strava activity":
                full_activities.append(activity)
            else:
                logger.debug(f"Skipping Strava stub activity: {activity.get('id')}")
        
        logger.info(f"Fetched {len(full_activities)} full activities (skipped {len(activities) - len(full_activities)} Strava stubs)")
        
        return full_activities
    
    def get_activities_since_last_import(
        self,
        last_import_date: datetime,
        batch_size: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch all activities since the last import date.
        
        Efficiently fetches new activities in batches to minimize API calls.
        
        Args:
            last_import_date: Date of last successful import
            batch_size: Number of activities to fetch per request
            
        Returns:
            List of new activity dictionaries
        """
        all_activities = []
        newest = None
        
        # Add 1 second to avoid duplicates
        oldest = last_import_date + timedelta(seconds=1)
        
        logger.info(f"Fetching activities since {oldest.strftime('%Y-%m-%d %H:%M:%S')}")
        
        while True:
            # Fetch batch
            activities = self.get_activities(
                oldest=oldest,
                newest=newest,
                limit=batch_size
            )
            
            if not activities:
                break
            
            all_activities.extend(activities)
            
            # If we got less than batch_size, we've fetched everything
            if len(activities) < batch_size:
                break
            
            # Set newest to the oldest activity in this batch for pagination
            # Activities are returned in descending date order
            last_activity_date = activities[-1].get("start_date_local")
            if last_activity_date:
                newest = datetime.fromisoformat(last_activity_date.replace("Z", "+00:00"))
                # Subtract 1 second to avoid missing activities with same timestamp
                newest = newest - timedelta(seconds=1)
            else:
                break
        
        logger.info(f"Fetched {len(all_activities)} new activities")
        return all_activities
    
    def get_athlete_info(self) -> Optional[Dict[str, Any]]:
        """Get athlete information.
        
        Returns:
            Athlete information dictionary or None if error
        """
        endpoint = f"/athlete/{self.athlete_id}"
        return self._make_request("GET", endpoint)
    
    def activities_to_dataframe(self, activities: List[Dict[str, Any]]) -> pl.DataFrame:
        """Convert activities list to Polars DataFrame.
        
        Preserves all ICU fields exactly as received.
        
        Args:
            activities: List of activity dictionaries from API
            
        Returns:
            Polars DataFrame with activities
        """
        if not activities:
            return pl.DataFrame()
        
        # Convert to DataFrame preserving all fields
        df = pl.DataFrame(activities)
        
        # Ensure required fields exist (add nulls if missing)
        required_fields = ["id", "start_date_local", "name"]
        for field in required_fields:
            if field not in df.columns:
                df = df.with_columns(pl.lit(None).alias(field))
        
        # Convert date strings to datetime if present
        if "start_date_local" in df.columns and df["start_date_local"].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col("start_date_local").str.to_datetime()
            )
        
        # Sort by date descending (most recent first)
        df = df.sort("start_date_local", descending=True)
        
        logger.info(f"Converted {len(df)} activities to DataFrame")
        
        return df


class IntervalsICUSyncManager:
    """Manages incremental sync with Intervals.icu."""
    
    def __init__(self, client: IntervalsICUClient, state_file: Path):
        """Initialize sync manager.
        
        Args:
            client: IntervalsICUClient instance
            state_file: Path to store sync state
        """
        self.client = client
        self.state_file = state_file
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load sync state from file."""
        if self.state_file.exists():
            import json
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {}
    
    def _save_state(self):
        """Save sync state to file."""
        import json
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def get_last_sync_date(self) -> Optional[datetime]:
        """Get the date of last successful sync."""
        if "last_sync_date" in self.state:
            return datetime.fromisoformat(self.state["last_sync_date"])
        return None
    
    def sync_activities(self, force_full: bool = False) -> pl.DataFrame:
        """Sync activities from Intervals.icu.
        
        Args:
            force_full: If True, fetch all activities (ignores last sync date)
            
        Returns:
            DataFrame with new activities
        """
        last_sync = self.get_last_sync_date()
        
        if force_full or last_sync is None:
            # Full sync: fetch last 365 days
            logger.info("Performing full sync (last 365 days)")
            oldest = datetime.now() - timedelta(days=365)
            activities = self.client.get_activities(oldest=oldest)
        else:
            # Incremental sync
            logger.info(f"Performing incremental sync since {last_sync}")
            activities = self.client.get_activities_since_last_import(last_sync)
        
        if activities:
            # Update last sync date to the most recent activity
            most_recent = max(activities, key=lambda x: x.get("start_date_local", ""))
            self.state["last_sync_date"] = most_recent["start_date_local"]
            self.state["last_sync_count"] = len(activities)
            self.state["last_sync_timestamp"] = datetime.now().isoformat()
            self._save_state()
            
            logger.info(f"Sync complete: {len(activities)} new activities")
        else:
            logger.info("No new activities to sync")
            self.state["last_sync_timestamp"] = datetime.now().isoformat()
            self.state["last_sync_count"] = 0
            self._save_state()
        
        return self.client.activities_to_dataframe(activities)
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status."""
        return {
            "last_sync_date": self.state.get("last_sync_date"),
            "last_sync_timestamp": self.state.get("last_sync_timestamp"),
            "last_sync_count": self.state.get("last_sync_count", 0),
            "athlete_id": self.client.athlete_id
        }