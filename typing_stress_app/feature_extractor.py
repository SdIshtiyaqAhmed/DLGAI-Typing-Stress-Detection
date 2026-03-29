import numpy as np

class FeatureExtractor:
    def __init__(self):
        """
        Initializes the feature extractor module.
        """
        pass

    def extract_features(self, keystrokes):
        """
        Extracts timing features from a list of captured keystrokes.
        """
        if len(keystrokes) < 2:
            return None, None
            
        # CRITICAL FIX: Sort keystrokes by press_time. 
        # Pynput records based on release order, which breaks sequence math for fast typists.
        keystrokes = sorted(keystrokes, key=lambda x: x['press_time'])
            
        hold_times = []
        ud_times = [] # Up-Down (Flight)
        dd_times = [] # Down-Down (Latency)
        error_count = 0
        
        # 1. Hold times and errors
        for stroke in keystrokes:
            hold_time = stroke['release_time'] - stroke['press_time']
            hold_times.append(max(hold_time, 0.0))
            
            key_name = stroke.get('key', '')
            if 'backspace' in key_name.lower() or 'delete' in key_name.lower():
                error_count += 1
            
        # 2. Sequential times
        for i in range(1, len(keystrokes)):
            # UD: Key N press - Key N-1 release (Standard Flight)
            ud = keystrokes[i]['press_time'] - keystrokes[i-1]['release_time']
            ud_times.append(max(ud, 0.0))
            
            # DD: Key N press - Key N-1 press (Latency)
            dd = keystrokes[i]['press_time'] - keystrokes[i-1]['press_time']
            dd_times.append(max(dd, 0.0))

        # --- 1. Math Heuristic Metrics (UI) ---
        mean_hold_time = np.mean(hold_times) if hold_times else 0.0
        std_hold_time = np.std(hold_times) if len(hold_times) > 1 else 0.0
        
        mean_flight_ud = np.mean(ud_times) if ud_times else 0.0
        std_flight_ud = np.std(ud_times) if len(ud_times) > 1 else 0.0
        
        # Total duration for UI speed display (KPS)
        total_duration = keystrokes[-1]['release_time'] - keystrokes[0]['press_time']
        total_duration = max(total_duration, 1.0)
        typing_speed_ui = len(keystrokes) / total_duration
        
        pause_count_ui = sum(1 for ud in ud_times if ud > 0.5)
        rhythm_variability = std_flight_ud
        error_rate = error_count / len(keystrokes)
        
        # Stability Score calculation
        instability_score = (rhythm_variability * 1.5) + (pause_count_ui * 0.2) + (mean_hold_time * 1.0)
        
        # Stability thresholds (Further Relaxed to 0.65 baseline)
        if instability_score < 0.65: 
            instability_level = "Stable Typing"
        elif instability_score <= 1.0:
            instability_level = "Moderate Instability"
        else:
            instability_level = "High Instability"
            
        metrics = {
            'mean_hold_time': mean_hold_time,
            'mean_flight_time': mean_flight_ud,
            'rhythm_variability': rhythm_variability,
            'pause_count': pause_count_ui,
            'typing_speed': typing_speed_ui,
            'instability_score': instability_score,
            'instability_level': instability_level
        }
        
        # --- 2. AI Model Feature Vector ---
        mean_flight_dd = np.mean(dd_times) if dd_times else 0.0
        std_flight_dd = np.std(dd_times) if len(dd_times) > 1 else 0.0
        
        # Pause count for AI: The notebook expects DD intervals. 
        # But to be safe, we use a higher threshold (0.6s) to reduce sensitivity.
        pause_count_ai = sum(1 for dd in dd_times if dd > 0.6)
        
        # Map speed back to the 20 / (flight_sum) reciprocal used in training
        typing_speed_ml = 20.0 / (mean_flight_ud * 11) if mean_flight_ud > 0.05 else 40.0

        feature_vector = [
            mean_hold_time, 
            std_hold_time,
            mean_flight_dd,
            std_flight_dd,
            typing_speed_ml,
            pause_count_ai,
            rhythm_variability,
            error_rate
        ]
        
        return np.array(feature_vector), metrics
