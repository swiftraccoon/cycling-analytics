"""Feature extractor for FIT file data to create comprehensive ML features."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import polars as pl
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)

class FITFeatureExtractor:
    """Extract comprehensive ML features from FIT file data."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.scaler = StandardScaler()
        warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    def extract_all_features(self, activity_df: pl.DataFrame) -> pl.DataFrame:
        """Extract all FIT-based features from activities DataFrame.
        
        Args:
            activity_df: DataFrame with FIT columns
            
        Returns:
            DataFrame with all extracted features
        """
        logger.info(f"Extracting FIT features from {len(activity_df)} activities")
        
        # Identify all FIT columns
        fit_columns = [col for col in activity_df.columns if col.startswith('fit_')]
        logger.info(f"Found {len(fit_columns)} FIT columns")
        
        # Extract features for each activity
        features_list = []
        
        for row in activity_df.iter_rows(named=True):
            features = dict(row)  # Keep all original columns
            
            # Extract basic FIT features
            basic_features = self._extract_basic_fit_features(row)
            features.update(basic_features)
            
            # Extract zone distribution features
            zone_features = self._extract_zone_features(row)
            features.update(zone_features)
            
            # Extract power analysis features
            power_features = self._extract_power_features(row)
            features.update(power_features)
            
            # Extract heart rate analysis features
            hr_features = self._extract_hr_features(row)
            features.update(hr_features)
            
            # Extract efficiency metrics
            efficiency_features = self._extract_efficiency_features(row)
            features.update(efficiency_features)
            
            # Extract variability metrics
            variability_features = self._extract_variability_features(row)
            features.update(variability_features)
            
            # Extract pedaling dynamics
            pedaling_features = self._extract_pedaling_features(row)
            features.update(pedaling_features)
            
            # Extract environmental features
            env_features = self._extract_environmental_features(row)
            features.update(env_features)
            
            # Extract device-specific features
            device_features = self._extract_device_features(row)
            features.update(device_features)
            
            # If we have time-series file, extract advanced features
            if row.get('fit_timeseries_file'):
                ts_features = self._extract_timeseries_features(row['fit_timeseries_file'])
                features.update(ts_features)
            
            features_list.append(features)
        
        result_df = pl.DataFrame(features_list)
        
        # Log feature extraction summary
        new_features = [col for col in result_df.columns if col not in activity_df.columns]
        logger.info(f"Extracted {len(new_features)} new ML features")
        
        return result_df
    
    def _extract_basic_fit_features(self, row: Dict) -> Dict[str, Any]:
        """Extract basic FIT features."""
        features = {}
        
        # Power metrics ratios
        if row.get('fit_power_normalized') and row.get('fit_power_avg'):
            features['ml_power_vi'] = row['fit_power_normalized'] / row['fit_power_avg'] if row['fit_power_avg'] > 0 else 0
        
        if row.get('fit_power_max') and row.get('fit_power_avg'):
            features['ml_power_max_ratio'] = row['fit_power_max'] / row['fit_power_avg'] if row['fit_power_avg'] > 0 else 0
        
        # Heart rate reserve utilization
        if row.get('fit_hr_avg') and row.get('fit_hr_max'):
            features['ml_hr_reserve_used'] = (row['fit_hr_avg'] / row['fit_hr_max']) * 100 if row['fit_hr_max'] > 0 else 0
        
        # Cadence consistency (inverse of std deviation)
        if row.get('fit_cadence_avg') and row.get('fit_ts_cadence_std'):
            features['ml_cadence_consistency'] = 1 / (1 + row['fit_ts_cadence_std'])
        
        # Altitude work (gain per km)
        if row.get('fit_altitude_gain') and row.get('distance'):
            features['ml_altitude_per_km'] = (row['fit_altitude_gain'] / (row['distance'] / 1000)) if row['distance'] > 0 else 0
        
        # Temperature adaptation stress
        if row.get('fit_temp_avg'):
            # Optimal temp is around 10-15C for cycling
            features['ml_temp_stress'] = abs(row['fit_temp_avg'] - 12.5)
        
        return features
    
    def _extract_zone_features(self, row: Dict) -> Dict[str, Any]:
        """Extract zone distribution features."""
        features = {}
        
        # HR zone features
        hr_zones = []
        for i in range(1, 6):
            zone_key = f'fit_hr_zones_zone{i}'
            if row.get(zone_key) is not None:
                hr_zones.append(row[zone_key])
        
        if hr_zones:
            features['ml_hr_zone_balance'] = np.std(hr_zones)  # Lower is more balanced
            features['ml_hr_zone_skew'] = stats.skew(hr_zones) if len(hr_zones) > 2 else 0
            features['ml_hr_zone_entropy'] = stats.entropy(hr_zones) if all(z >= 0 for z in hr_zones) else 0
            
            # Polarization index (Z1+Z5 vs Z2+Z3+Z4)
            if len(hr_zones) >= 5:
                low_high = hr_zones[0] + hr_zones[4]
                middle = sum(hr_zones[1:4])
                features['ml_hr_polarization'] = low_high / (middle + 0.01)
        
        # Power zone features
        power_zones = []
        for i in range(1, 8):
            zone_key = f'fit_power_zones_zone{i}'
            if row.get(zone_key) is not None:
                power_zones.append(row[zone_key])
        
        if power_zones:
            features['ml_power_zone_balance'] = np.std(power_zones)
            features['ml_power_zone_skew'] = stats.skew(power_zones) if len(power_zones) > 2 else 0
            features['ml_power_zone_entropy'] = stats.entropy(power_zones) if all(z >= 0 for z in power_zones) else 0
            
            # Training impulse zones
            if len(power_zones) >= 7:
                # Aerobic vs anaerobic balance
                aerobic = sum(power_zones[:3])  # Z1-Z3
                anaerobic = sum(power_zones[3:])  # Z4-Z7
                features['ml_aerobic_anaerobic_ratio'] = aerobic / (anaerobic + 0.01)
                
                # Sweet spot time (Z3+Z4)
                features['ml_sweet_spot_pct'] = power_zones[2] + power_zones[3] if len(power_zones) > 3 else 0
                
                # High intensity time (Z5+Z6+Z7)
                features['ml_high_intensity_pct'] = sum(power_zones[4:]) if len(power_zones) > 4 else 0
        
        return features
    
    def _extract_power_features(self, row: Dict) -> Dict[str, Any]:
        """Extract advanced power features."""
        features = {}
        
        # Power curve characteristics
        if row.get('fit_power_normalized'):
            np_val = row['fit_power_normalized']
            
            # Coggan's training levels
            if row.get('fit_session_threshold_power'):
                ftp = row['fit_session_threshold_power']
                features['ml_if'] = np_val / ftp if ftp > 0 else 0  # Intensity Factor
                
                # Training zone based on IF
                if features['ml_if'] < 0.55:
                    features['ml_training_level'] = 1  # Recovery
                elif features['ml_if'] < 0.75:
                    features['ml_training_level'] = 2  # Endurance
                elif features['ml_if'] < 0.85:
                    features['ml_training_level'] = 3  # Tempo
                elif features['ml_if'] < 0.95:
                    features['ml_training_level'] = 4  # Threshold
                elif features['ml_if'] < 1.05:
                    features['ml_training_level'] = 5  # VO2max
                else:
                    features['ml_training_level'] = 6  # Anaerobic
        
        # Power distribution metrics from time-series features
        if row.get('fit_ts_power_std'):
            features['ml_power_variability'] = row['fit_ts_power_std']
        
        if row.get('fit_ts_power_smoothness'):
            features['ml_power_smoothness'] = row['fit_ts_power_smoothness']
        
        # Quadrant analysis (force vs velocity)
        if row.get('fit_power_avg') and row.get('fit_cadence_avg'):
            # Approximate force from power and cadence
            features['ml_avg_force'] = (row['fit_power_avg'] * 60) / (row['fit_cadence_avg'] * 2 * np.pi * 0.1725) if row['fit_cadence_avg'] > 0 else 0
        
        # W/kg if weight available
        if row.get('fit_power_avg') and row.get('icu_weight'):
            features['ml_power_to_weight'] = row['fit_power_avg'] / row['icu_weight'] if row['icu_weight'] > 0 else 0
        
        if row.get('fit_power_normalized') and row.get('icu_weight'):
            features['ml_np_to_weight'] = row['fit_power_normalized'] / row['icu_weight'] if row['icu_weight'] > 0 else 0
        
        # Critical power metrics
        for threshold in [100, 150, 200, 250, 300, 350, 400, 450, 500]:
            key = f'fit_ts_time_above_{threshold}w_pct'
            if row.get(key):
                features[f'ml_time_above_{threshold}w'] = row[key]
        
        return features
    
    def _extract_hr_features(self, row: Dict) -> Dict[str, Any]:
        """Extract heart rate features."""
        features = {}
        
        # HR efficiency (power per beat)
        if row.get('fit_power_avg') and row.get('fit_hr_avg'):
            features['ml_power_per_beat'] = row['fit_power_avg'] / row['fit_hr_avg'] if row['fit_hr_avg'] > 0 else 0
        
        if row.get('fit_power_normalized') and row.get('fit_hr_avg'):
            features['ml_np_per_beat'] = row['fit_power_normalized'] / row['fit_hr_avg'] if row['fit_hr_avg'] > 0 else 0
        
        # HR drift indicator
        if row.get('fit_ts_hr_drift'):
            features['ml_hr_drift'] = row['fit_ts_hr_drift']
        
        # Aerobic decoupling
        if row.get('fit_ts_hr_decoupling'):
            features['ml_aerobic_decoupling'] = row['fit_ts_hr_decoupling']
        
        # HR recovery metrics
        if row.get('icu_hrrc'):
            features['ml_hr_recovery'] = row['icu_hrrc']
        
        # HR variability coefficient
        if row.get('fit_ts_hr_cv'):
            features['ml_hr_variability_coef'] = row['fit_ts_hr_cv']
        
        # Training load from HR
        if row.get('hr_load'):
            features['ml_hr_training_load'] = row['hr_load']
        
        return features
    
    def _extract_efficiency_features(self, row: Dict) -> Dict[str, Any]:
        """Extract efficiency metrics."""
        features = {}
        
        # Gross efficiency (work / energy)
        if row.get('fit_session_total_work') and row.get('fit_session_total_calories'):
            # Convert calories to joules (1 cal = 4.184 J)
            energy_joules = row['fit_session_total_calories'] * 4184
            features['ml_gross_efficiency'] = (row['fit_session_total_work'] / energy_joules * 100) if energy_joules > 0 else 0
        
        # Speed per watt
        if row.get('fit_speed_avg') and row.get('fit_power_avg'):
            features['ml_speed_per_watt'] = row['fit_speed_avg'] / row['fit_power_avg'] if row['fit_power_avg'] > 0 else 0
        
        # VAM (Vertical Ascent in Meters per hour)
        if row.get('fit_altitude_gain') and row.get('fit_session_total_timer_time'):
            hours = row['fit_session_total_timer_time'] / 3600 if row['fit_session_total_timer_time'] else 0
            features['ml_vam'] = row['fit_altitude_gain'] / hours if hours > 0 else 0
        
        # Power to HR ratio (fitness indicator)
        if row.get('fit_power_avg') and row.get('fit_hr_avg'):
            features['ml_power_hr_ratio'] = row['fit_power_avg'] / row['fit_hr_avg'] if row['fit_hr_avg'] > 0 else 0
        
        # Normalized power to HR ratio
        if row.get('fit_power_normalized') and row.get('fit_hr_avg'):
            features['ml_np_hr_ratio'] = row['fit_power_normalized'] / row['fit_hr_avg'] if row['fit_hr_avg'] > 0 else 0
        
        return features
    
    def _extract_variability_features(self, row: Dict) -> Dict[str, Any]:
        """Extract variability and consistency metrics."""
        features = {}
        
        # Power variability index
        if row.get('fit_power_variability_index'):
            features['ml_vi'] = row['fit_power_variability_index']
        
        # Coefficient of variation for different metrics
        cv_metrics = [
            ('power', 'fit_ts_power_cv'),
            ('hr', 'fit_ts_hr_cv'),
            ('cadence', 'fit_ts_cadence_cv'),
        ]
        
        for name, key in cv_metrics:
            if row.get(key):
                features[f'ml_{name}_cv'] = row[key]
        
        # Smoothness indices
        if row.get('fit_ts_power_smoothness'):
            features['ml_power_smooth_index'] = 1 / (1 + row['fit_ts_power_smoothness'])
        
        # Distribution characteristics
        dist_metrics = [
            ('power_skewness', 'fit_ts_power_skewness'),
            ('power_kurtosis', 'fit_ts_power_kurtosis'),
            ('power_iqr', 'fit_ts_power_iqr'),
        ]
        
        for name, key in dist_metrics:
            if row.get(key):
                features[f'ml_{name}'] = row[key]
        
        return features
    
    def _extract_pedaling_features(self, row: Dict) -> Dict[str, Any]:
        """Extract pedaling dynamics features."""
        features = {}
        
        # Pedal balance
        if row.get('fit_session_left_right_balance'):
            balance = row['fit_session_left_right_balance']
            features['ml_pedal_balance'] = balance
            features['ml_pedal_asymmetry'] = abs(50 - balance)
        
        # Torque effectiveness
        if row.get('fit_session_avg_left_torque_effectiveness'):
            features['ml_left_torque_eff'] = row['fit_session_avg_left_torque_effectiveness']
        
        if row.get('fit_session_avg_right_torque_effectiveness'):
            features['ml_right_torque_eff'] = row['fit_session_avg_right_torque_effectiveness']
        
        # Pedal smoothness
        if row.get('fit_session_avg_left_pedal_smoothness'):
            features['ml_left_pedal_smooth'] = row['fit_session_avg_left_pedal_smoothness']
        
        if row.get('fit_session_avg_right_pedal_smoothness'):
            features['ml_right_pedal_smooth'] = row['fit_session_avg_right_pedal_smoothness']
        
        # Combined pedaling efficiency
        te_vals = []
        if row.get('fit_session_avg_left_torque_effectiveness'):
            te_vals.append(row['fit_session_avg_left_torque_effectiveness'])
        if row.get('fit_session_avg_right_torque_effectiveness'):
            te_vals.append(row['fit_session_avg_right_torque_effectiveness'])
        
        if te_vals:
            features['ml_avg_torque_effectiveness'] = np.mean(te_vals)
        
        ps_vals = []
        if row.get('fit_session_avg_left_pedal_smoothness'):
            ps_vals.append(row['fit_session_avg_left_pedal_smoothness'])
        if row.get('fit_session_avg_right_pedal_smoothness'):
            ps_vals.append(row['fit_session_avg_right_pedal_smoothness'])
        
        if ps_vals:
            features['ml_avg_pedal_smoothness'] = np.mean(ps_vals)
        
        # Cadence-power relationship
        if row.get('fit_cadence_avg') and row.get('fit_power_avg'):
            features['ml_power_per_rpm'] = row['fit_power_avg'] / row['fit_cadence_avg'] if row['fit_cadence_avg'] > 0 else 0
        
        return features
    
    def _extract_environmental_features(self, row: Dict) -> Dict[str, Any]:
        """Extract environmental and conditions features."""
        features = {}
        
        # Temperature effects
        if row.get('fit_temp_avg'):
            temp = row['fit_temp_avg']
            features['ml_temp_avg'] = temp
            
            # Temperature zones
            if temp < 0:
                features['ml_temp_zone'] = 1  # Freezing
            elif temp < 10:
                features['ml_temp_zone'] = 2  # Cold
            elif temp < 20:
                features['ml_temp_zone'] = 3  # Optimal
            elif temp < 30:
                features['ml_temp_zone'] = 4  # Warm
            else:
                features['ml_temp_zone'] = 5  # Hot
        
        # Temperature range (adaptation stress)
        if row.get('fit_ts_temp_range'):
            features['ml_temp_variation'] = row['fit_ts_temp_range']
        
        # Altitude effects
        if row.get('fit_session_avg_altitude'):
            alt = row['fit_session_avg_altitude']
            features['ml_avg_altitude'] = alt
            
            # Altitude zones (effects on performance)
            if alt < 500:
                features['ml_altitude_zone'] = 1  # Sea level
            elif alt < 1500:
                features['ml_altitude_zone'] = 2  # Low altitude
            elif alt < 2500:
                features['ml_altitude_zone'] = 3  # Moderate altitude
            else:
                features['ml_altitude_zone'] = 4  # High altitude
        
        # Grade variations
        if row.get('fit_ts_altitude_change_rate'):
            features['ml_terrain_variability'] = row['fit_ts_altitude_change_rate']
        
        return features
    
    def _extract_device_features(self, row: Dict) -> Dict[str, Any]:
        """Extract device and sensor features."""
        features = {}
        
        # Device type indicators
        if row.get('fit_device_manufacturer'):
            features['ml_device_manufacturer'] = hash(str(row['fit_device_manufacturer'])) % 100
        
        if row.get('fit_device_product'):
            features['ml_device_product'] = hash(str(row['fit_device_product'])) % 100
        
        # Data quality indicators
        if row.get('fit_data_points'):
            features['ml_data_density'] = row['fit_data_points']
        
        if row.get('fit_lap_count'):
            features['ml_lap_count'] = row['fit_lap_count']
        
        if row.get('fit_event_count'):
            features['ml_event_count'] = row['fit_event_count']
        
        # Sensor availability flags
        features['ml_has_power'] = 1 if row.get('device_watts') else 0
        features['ml_has_hr'] = 1 if row.get('has_heartrate') else 0
        features['ml_has_cadence'] = 1 if row.get('fit_cadence_avg') else 0
        features['ml_has_temperature'] = 1 if row.get('fit_temp_avg') else 0
        
        return features
    
    def _extract_timeseries_features(self, timeseries_file: str) -> Dict[str, Any]:
        """Extract features from time-series parquet file."""
        features = {}
        
        ts_path = Path(timeseries_file)
        if not ts_path.exists():
            return features
        
        try:
            # Load time-series data
            ts_df = pl.read_parquet(ts_path)
            
            if ts_df.is_empty():
                return features
            
            # Extract advanced time-series features
            
            # Power spectral density features
            if 'power' in ts_df.columns:
                power_data = ts_df['power'].drop_nulls().to_numpy()
                if len(power_data) > 10:
                    # Frequency domain analysis
                    freqs, psd = signal.periodogram(power_data, fs=1.0)
                    
                    # Find dominant frequency
                    features['ml_power_dominant_freq'] = freqs[np.argmax(psd)]
                    
                    # Spectral entropy
                    psd_norm = psd / np.sum(psd)
                    features['ml_power_spectral_entropy'] = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
                    
                    # Autocorrelation features
                    autocorr = np.correlate(power_data - np.mean(power_data), 
                                          power_data - np.mean(power_data), mode='full')
                    autocorr = autocorr[len(autocorr)//2:]
                    autocorr = autocorr / autocorr[0]
                    
                    # Find first minimum (pedaling cycle)
                    if len(autocorr) > 60:
                        features['ml_power_autocorr_period'] = np.argmin(autocorr[:60])
            
            # Heart rate complexity
            if 'heart_rate' in ts_df.columns:
                hr_data = ts_df['heart_rate'].drop_nulls().to_numpy()
                if len(hr_data) > 10:
                    # Sample entropy (complexity measure)
                    features['ml_hr_sample_entropy'] = self._sample_entropy(hr_data)
                    
                    # Detrended fluctuation analysis
                    features['ml_hr_dfa'] = self._dfa(hr_data)
            
            # Cadence stability
            if 'cadence' in ts_df.columns:
                cadence_data = ts_df.filter(pl.col('cadence') > 0)['cadence'].to_numpy()
                if len(cadence_data) > 10:
                    # Coefficient of variation
                    features['ml_cadence_stability'] = 1 / (1 + stats.variation(cadence_data))
                    
                    # Consecutive differences
                    diffs = np.diff(cadence_data)
                    features['ml_cadence_smoothness'] = 1 / (1 + np.std(diffs))
            
            # Multi-signal coupling
            if 'power' in ts_df.columns and 'heart_rate' in ts_df.columns:
                # Filter to valid data
                valid_df = ts_df.filter(
                    (pl.col('power').is_not_null()) & 
                    (pl.col('heart_rate').is_not_null())
                )
                
                if len(valid_df) > 10:
                    power = valid_df['power'].to_numpy()
                    hr = valid_df['heart_rate'].to_numpy()
                    
                    # Cross-correlation
                    correlation = np.corrcoef(power, hr)[0, 1]
                    features['ml_power_hr_correlation'] = correlation
                    
                    # Coherence
                    if len(power) > 100:
                        f, coh = signal.coherence(power, hr, fs=1.0)
                        features['ml_power_hr_coherence'] = np.mean(coh)
            
            # Fatigue indicators
            if 'elapsed_time' in ts_df.columns and 'power' in ts_df.columns:
                valid_df = ts_df.filter(pl.col('power').is_not_null())
                if len(valid_df) > 100:
                    # Split into quarters
                    quarter = len(valid_df) // 4
                    if quarter > 0:
                        q1_power = valid_df[:quarter]['power'].mean()
                        q4_power = valid_df[-quarter:]['power'].mean()
                        
                        if q1_power > 0:
                            features['ml_power_fade'] = ((q1_power - q4_power) / q1_power) * 100
                
                # Similar for HR
                if 'heart_rate' in ts_df.columns:
                    valid_df = ts_df.filter(pl.col('heart_rate').is_not_null())
                    if len(valid_df) > 100:
                        quarter = len(valid_df) // 4
                        if quarter > 0:
                            q1_hr = valid_df[:quarter]['heart_rate'].mean()
                            q4_hr = valid_df[-quarter:]['heart_rate'].mean()
                            
                            if q1_hr > 0:
                                features['ml_hr_rise'] = ((q4_hr - q1_hr) / q1_hr) * 100
            
        except Exception as e:
            logger.warning(f"Error extracting time-series features: {e}")
        
        return features
    
    def _sample_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy of time series."""
        N = len(data)
        if N < m + 1:
            return 0
        
        r = r * np.std(data)
        
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        
        def _phi(m):
            patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
            C = 0
            for i in range(len(patterns)):
                template = patterns[i]
                for j in range(len(patterns)):
                    if i != j and _maxdist(template, patterns[j]) <= r:
                        C += 1
            return C / (N - m + 1) / (N - m) if (N - m) > 0 else 0
        
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        
        if phi_m1 == 0:
            return 0
        
        return -np.log(phi_m1 / phi_m) if phi_m > 0 else 0
    
    def _dfa(self, data: np.ndarray) -> float:
        """Detrended Fluctuation Analysis."""
        N = len(data)
        if N < 16:
            return 0
        
        # Integrate the signal
        y = np.cumsum(data - np.mean(data))
        
        # Calculate fluctuation for different box sizes
        scales = np.logspace(0.5, np.log10(N/4), 10, dtype=int)
        flucts = []
        
        for scale in scales:
            if scale < 4:
                continue
                
            # Calculate fluctuation at this scale
            n_segments = N // scale
            if n_segments < 1:
                continue
                
            variance = []
            for i in range(n_segments):
                segment = y[i*scale:(i+1)*scale]
                if len(segment) >= 2:
                    x = np.arange(len(segment))
                    coeffs = np.polyfit(x, segment, 1)
                    fit = np.polyval(coeffs, x)
                    variance.append(np.mean((segment - fit) ** 2))
            
            if variance:
                flucts.append(np.sqrt(np.mean(variance)))
        
        if len(flucts) < 2:
            return 0
        
        # Fit log-log plot
        coeffs = np.polyfit(np.log(scales[:len(flucts)]), np.log(flucts), 1)
        
        return coeffs[0]  # Scaling exponent