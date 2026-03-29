import sys
import subprocess

def install_dependencies():
    """Automatically installs necessary libraries if they are not already available on the system."""
    dependencies = {
        'pynput': 'pynput',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'tensorflow': 'tensorflow',
        'sklearn': 'scikit-learn',
        'joblib': 'joblib',
        'streamlit': 'streamlit' # Not strictly required for the CLI, but included for the optional bonus Streamlit UI
    }
    
    for module_name, pip_name in dependencies.items():
        try:
            __import__(module_name)
        except ImportError:
            print(f"Installing missing dependency: {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

# We run dependency installation immediately upon launching the script
print("Checking and installing dependencies...")
install_dependencies()

# Once dependencies are available, we can safely import our local modules
from typing_capture import TypingCapture
from feature_extractor import FeatureExtractor
from predictor import Predictor

def main():
    print("=" * 60)
    print("      EARLY STRESS DETECTION FROM TYPING PATTERNS")
    print("=" * 60)
    print("\nInitializing machine learning components... (This may take a moment)")
    
    # Initialize the inference pipeline modules.
    # The models are expected to be in the current directory, or the parent folder.
    predictor_engine = Predictor(model_dir='.')
    feat_extractor = FeatureExtractor()
    
    # We will record 10 seconds of typing pattern data
    capture_duration = 10 
    capture_engine = TypingCapture(duration=capture_duration)
    
    print("\n" + "-" * 60)
    print(f"Get ready to type! The application will record your typing for {capture_duration} seconds.")
    print("To get a realistic sample, type a natural paragraph of text or code.")
    print("-" * 60)
    
    input("Press ENTER when you are ready to begin capturing...\n")
    
    try:
        # STEP 1: Real-time Keystroke Capture
        keystrokes = capture_engine.start_capture()
        
        # STEP 2: Extraction of Keystroke Timing Features
        print("\nExtracting timing features from your keystrokes...")
        features, metrics = feat_extractor.extract_features(keystrokes)
        
        if features is None:
            print("Capture failed: Could not extract features. Ensure you typed during the window.")
            return
            
        print(f"Calculated Feature Vector (8 features):\n{features}\n")
        
        # STEP 3 & 4: Model Loading & Inference
        print("Analyzing typing pattern...\n")
        predictions = predictor_engine.predict(features)
        
        # Format the Agreement Output
        rf_class = predictions['rf_class'] if predictions['rf_class'] else "N/A"
        rf_conf = f"{predictions['rf_conf']:.0%}" if predictions['rf_class'] else "N/A"
        nn_class = predictions['nn_class'] if predictions['nn_class'] else "N/A"
        nn_conf = f"{predictions['nn_conf']:.0%}" if predictions['nn_class'] else "N/A"
        
        agreement_text = "YES" if predictions.get('agreement', False) else "NO"
        
        # STEP 5: Final Output
        st_lvl = metrics['instability_level']
        
        print("\n" + "=" * 60)
        print("                 TYPING ANALYSIS SUMMARY")
        print("=" * 60)
        
        if st_lvl == "Stable Typing":
            print("STATUS: CALM & CONSISTENT")
            print("Your typing rhythm is steady, indicating a focused and relaxed state.")
        elif st_lvl == "Moderate Instability":
            print("STATUS: MODERATE FLUCTUATIONS")
            print("Some minor variations in rhythm detected. Could be normal fatigue or focus shifts.")
        else:
            print("STATUS: SIGNIFICANT INSTABILITY")
            print("Highly irregular patterns detected. This often correlates with stress or frustration.")
            
        print("-" * 60)
        
        # High level summary
        speed = metrics['typing_speed']
        speed_text = "Fast" if speed > 6 else "Moderate" if speed >= 3 else "Deliberate"
        rhythm = metrics['rhythm_variability']
        rhythm_text = "Excellent" if rhythm < 0.1 else "Steady" if rhythm < 0.22 else "Variable"
        
        print(f"Pace:   {speed_text:<15} ({speed:.1f} keys/sec)")
        print(f"Flow:   {rhythm_text:<15} (Var: {rhythm:.2f})")
        print(f"Hold:   {metrics['mean_hold_time']:.2f}s average per key")
        print("-" * 60)

        # Advanced Section (Optional view)
        show_tech = input("\nWould you like to see technical ML model data? (y/n): ").lower()
        if show_tech == 'y':
            print("\n" + "." * 30 + " ADVANCED TECHNICAL DATA " + "." * 30)
            print(f"Random Forest Prediction: Class {rf_class} ({rf_conf} confidence)")
            
            nn_disp = nn_class if nn_class else "OFFLINE"
            nn_conf_disp = f"{nn_conf}" if nn_class else "ERROR"
            print(f"Neural Network Prediction: Class {nn_disp} ({nn_conf_disp} confidence)")
            
            agreement = predictions.get('agreement')
            if agreement is True:
                agreement_text = "YES (Confirmed)"
            elif agreement is False:
                agreement_text = "NO (Disagreement)"
            else:
                agreement_text = "N/A (Neural Network failed to load)"
                
            print(f"Model Agreement: {agreement_text}")
            print(f"Calculated Instability Score: {metrics['instability_score']:.2f}")
            print("." * 85 + "\n")
        
        print("Thank you for using the Keyboard Stress Detection tool.")
        print("=" * 60 + "\n")
        
    except KeyboardInterrupt:
        # Graceful handling if the user presses Ctrl+C
        print("\nProcess interrupted by user. Exiting the application.")
    except Exception as e:
        print(f"\nAn unexpected runtime error occurred: {e}")

if __name__ == "__main__":
    main()
