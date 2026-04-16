import streamlit as st
import pandas as pd
import threading
import time
import sys
import os
import importlib.metadata

# Ensure the app's own directory is always on sys.path so sibling modules
# (typing_capture, feature_extractor, predictor) can be imported regardless
# of which directory the user launches Streamlit from.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

# FIX for Python 3.9 + TF 2.16+ where Keras/TensorFlow looks for 'packages_distributions' 
# which was only added to importlib.metadata in Python 3.10.
if sys.version_info < (3, 10):
    try:
        import importlib_metadata
        # Replace the module in sys.modules so any library doing 
        # 'from importlib.metadata import ...' gets the backport instead.
        sys.modules['importlib.metadata'] = importlib_metadata
    except ImportError:
        # Fallback dummy if backport is missing
        if not hasattr(importlib.metadata, 'packages_distributions'):
            def dummy_packages_distributions(): return {}
            importlib.metadata.packages_distributions = dummy_packages_distributions

try:
    from typing_capture import TypingCapture
    from feature_extractor import FeatureExtractor
    from predictor import Predictor
except ImportError:
    st.error("Could not import required local modules. Please ensure this script is run from the `typing_stress_app` folder.")
    st.stop()

# Initialize Streamlit UI configuration
st.set_page_config(page_title="Keyboard Stress Detection", page_icon="🤖", layout="centered")

@st.cache_resource
def load_ml_models():
    """Cache models so we don't reload tensorflow/scikit-learn on every UI interaction."""
    return Predictor(model_dir='.'), FeatureExtractor()

try:
    predictor, extractor = load_ml_models()
except Exception as e:
    st.error(f"Error loading Machine Learning Models: {e}")
    st.stop()

# Streamlit App Session States
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'keystrokes' not in st.session_state:
    st.session_state.keystrokes = []
if 'features' not in st.session_state:
    st.session_state.features = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'is_capturing' not in st.session_state:
    st.session_state.is_capturing = False

# Typing Pattern Analysis
st.markdown("### Analysis of Typing Rhythm")
st.markdown("This tool checks your typing rhythm to see if your patterns remain steady or change over time. It compares your typing to a math-based logic and an AI model to give a final result.")

with st.expander("How the Analysis Works"):
    st.markdown("""
    ### 1. Recording Data
    The tool records the exact timing of every key you press and release. We do not record the words you type, only the time between them.
    
    ### 2. Finding Patterns
    We calculate two main things:
    - **Hold Time:** How long you hold down each key.
    - **Gap Time:** The time between releasing one key and pressing the next.
    
    ### 3. Comparing Results
    Your data is checked by two systems:
    - **Math Logic:** A standard calculation that looks for rhythm changes.
    - **AI Model:** A neural network that has learned common typing patterns.
    
    ### 4. Final Summary
    The system looks at both results together to give you a single, clear answer.
    """)

st.divider()

# Dashboard Settings
st.subheader("1. Setup")
duration = st.slider("Typing Test Time (seconds):", min_value=5, max_value=30, value=10, step=1, help="The total time you will spend typing for the test.")
prep_delay = st.slider("Wait Time Before Start (seconds):", min_value=0, max_value=10, value=3, step=1, help="Time to get your hands ready on the keyboard.")

st.subheader("2. Start Typing")
col_ctrl, col_type = st.columns([1, 1.5])

with col_ctrl:
    st.info(f"Test will run for **{duration} seconds**.")
    if st.button("Start Typing Now", disabled=st.session_state.recording, type="primary", use_container_width=True):
        st.session_state.recording = True
        st.rerun()

with col_type:
    st.text_area(
        "Typing Box", 
        placeholder="Click start and then type here..." if not st.session_state.recording else "TYPE HERE NOW", 
        height=120, 
        disabled=not st.session_state.recording,
        help="Type any sentence here during the test."
    )

if st.session_state.recording:
    if prep_delay > 0:
        countdown_zone = st.empty()
        for i in range(prep_delay, 0, -1):
            countdown_zone.markdown(f"<div style='text-align:center; padding: 20px; border-radius: 10px; background: #f8f9fa; border: 1px solid #dee2e6;'><h3 style='margin:0;'>Get Ready: <span style='color: #28a745;'>{i}</span></h3></div>", unsafe_allow_html=True)
            time.sleep(1)
        countdown_zone.empty()

    with st.spinner(f"RECORDING... PLEASE TYPE ({duration}s)"):
        capture_engine = TypingCapture(duration=duration)
        # starts the listener and sleeps for the duration synchronously
        st.session_state.keystrokes = capture_engine.start_capture() 
    
    # Process results immediately after capture
    if st.session_state.keystrokes:
        features, metrics = extractor.extract_features(st.session_state.keystrokes)
        st.session_state.features = features
        st.session_state.metrics = metrics
        
        if st.session_state.features is not None:
            predictions = predictor.predict(st.session_state.features)
            st.session_state.prediction = predictions
    
    # Reset recording state and refresh UI to show results
    st.session_state.recording = False
    st.rerun()

st.divider()

# Displays
if st.session_state.features is not None:
    st.header("3. Results Analysis")
    
    # --- USER FRIENDLY SECTION ---
    st.subheader("Summary")
    
    # Determine the status theme based on instability level
    lvl = st.session_state.metrics['instability_level']
    nn_class = st.session_state.prediction['nn_class'] # "0"=Normal, "1"=Stress
    rf_class = st.session_state.prediction['rf_class']

    # --- CONSENSUS ENGINE ---
    # We use a trio of experts: Math Heuristic, Neural Network, and Random Forest.
    h_vote = 0 if lvl == "Stable Typing" else 1
    nn_vote = int(nn_class) if nn_class in ["0", "1"] else h_vote
    rf_vote = int(rf_class) if rf_class in ["0", "1"] else h_vote

    # Final Decision Logic: Majority Vote (at least 2 systems must agree)
    votes = [h_vote, nn_vote, rf_vote]
    stress_votes = votes.count(1)
    
    if stress_votes >= 2:
        final_status = "STRESS"
        final_color = "error"
        final_msg = "The system detected changes in your typing rhythm consistent with stress or cognitive load."
    else:
        final_status = "NORMAL"
        final_color = "success"
        final_msg = "Your typing rhythm is consistent and stable."

    # Final Summary Card
    st.markdown(f"### Final Result: <span style='color:{'#dc3545' if final_status == 'STRESS' else '#28a745' if final_status == 'NORMAL' else '#6c757d'}'>{final_status}</span>", unsafe_allow_html=True)
    if getattr(st, final_color): getattr(st, final_color)(f"**{final_msg}**")
    
    # Transparency Indicator
    agreement_count = votes.count(1) if final_status == "STRESS" else votes.count(0)
    st.caption(f"Reasoning: {agreement_count}/3 systems agreed on this result.")


    # High-level summary cards (Clearer than raw metrics)
    col1, col2, col3 = st.columns(3)
    
    # Evaluate speed (roughly: < 3 slow, 3-6 moderate, > 6 fast)
    speed = st.session_state.metrics['typing_speed']
    speed_text = "Fast" if speed > 6 else "Moderate" if speed >= 3 else "Deliberate"
    
    # Evaluate rhythm (variability < 0.1 excellent, < 0.2 good, else erratic)
    rhythm = st.session_state.metrics['rhythm_variability']
    rhythm_text = "Excellent" if rhythm < 0.1 else "Steady" if rhythm < 0.22 else "Variable"

    # Evaluate Hold Time (standard is around 0.1-0.2s)
    hold = st.session_state.metrics['mean_hold_time']
    hold_text = "Light" if hold < 0.1 else "Standard" if hold < 0.18 else "Heavy"

    with col1:
        st.markdown(f"**Typing Speed**  \n### {speed_text}  \n_{speed:.1f} CPS_")
    with col2:
        st.markdown(f"**Rhythm**  \n### {rhythm_text}  \n_Var: {rhythm:.2f}_")
    with col3:
        st.markdown(f"**Hold Time**  \n### {hold_text}  \n_{hold:.2f}s Avg._")

    st.divider()

    # --- TECHNICAL / ADVANCED SECTION ---
    with st.expander("Show Technical Details"):
        # Model Result Cards
        st.subheader("Results from Individual Systems")
        st.caption("A breakdown of what our analysis systems found.")
        col1, col2, col3 = st.columns(3)
        
        # Model-Specific Label Mapping
        MAP_LABEL = { "0": "Normal", "1": "Stress" }
        
        nn_class = st.session_state.prediction['nn_class']
        rf_class = st.session_state.prediction['rf_class']
        
        # Math Logic formatting
        h_val = "Normal" if h_vote == 0 else "Stress"
        h_class_style = "status-normal" if h_vote == 0 else "status-stress"
        
        # NN formatting
        nn_val = MAP_LABEL.get(nn_class, "Unavailable")
        nn_conf = f"{st.session_state.prediction['nn_conf'] * 100:.1f}%" if nn_class else "Analysis Error"
        nn_class_style = "status-normal" if nn_class == "0" else "status-stress" if nn_class == "1" else "status-gray"

        # RF formatting
        rf_val = MAP_LABEL.get(rf_class, "Unavailable")
        rf_conf = f"{st.session_state.prediction['rf_conf'] * 100:.1f}%" if rf_class else "Analysis Error"
        rf_class_style = "status-normal" if rf_class == "0" else "status-stress" if rf_class == "1" else "status-gray"
        
        # System Sync (Agreement now checks all three)
        all_votes = [h_vote, int(nn_class) if nn_class else None, int(rf_class) if rf_class else None]
        all_votes = [v for v in all_votes if v is not None]
        is_consensus = len(set(all_votes)) == 1 if all_votes else False
        
        agreement_text = "FULL AGREEMENT" if is_consensus else "DIVERGENT RESULTS"
        agreement_style = "status-normal" if is_consensus else "status-gray"

        # Custom CSS for Professional Minimalist Theme
        st.markdown("""
            <style>
            .result-container {
                display: flex;
                gap: 20px;
                margin-top: 25px;
                flex-wrap: wrap;
            }
            .result-card {
                background: white;
                border: 1px solid #e9ecef;
                border-radius: 12px;
                padding: 20px;
                flex: 1;
                min-width: 200px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.04);
            }
            .result-label {
                text-transform: uppercase;
                letter-spacing: 0.05rem;
                font-size: 0.7rem;
                color: #6c757d;
                font-weight: 600;
                margin-bottom: 8px;
            }
            .result-value {
                font-size: 1.4rem;
                font-weight: 700;
                margin-bottom: 5px;
            }
            .result-conf {
                font-size: 0.8rem;
                color: #adb5bd;
            }
            .status-normal { color: #28a745; }
            .status-stress { color: #dc3545; }
            .status-gray { color: #6c757d; }
            </style>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="result-container">
                <div class="result-card">
                    <div class="result-label">Math Logic</div>
                    <div class="result-value {h_class_style}">{h_val}</div>
                    <div class="result-conf">Rhythm calculation</div>
                </div>
                <div class="result-card">
                    <div class="result-label">Neural Network</div>
                    <div class="result-value {nn_class_style}">{nn_val}</div>
                    <div class="result-conf">Conf: {nn_conf}</div>
                </div>
                <div class="result-card">
                    <div class="result-label">Random Forest</div>
                    <div class="result-value {rf_class_style}">{rf_val}</div>
                    <div class="result-conf">Conf: {rf_conf}</div>
                </div>
            </div>
            <div style="margin-top: 20px; padding: 12px; border-radius: 8px; background: #f8f9fa; border: 1px solid #e9ecef;">
                <span class="result-label">Agreement Check:</span> 
                <span class="result-value {agreement_style}" style="font-size: 1.1rem; margin-left: 10px;">{agreement_text}</span>
            </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 **Interpretation:** Class **Normal** indicates a relaxed, consistent typing flow. Class **Stress** indicates rhythm instability or erratic timing often linked to heightened cognitive load or anxiety.")
        
        st.subheader("Raw Typing Metrics")
        st.caption("Specific timing measurements extracted from the raw keystroke stream.")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(label="Hold Time", value=f"{st.session_state.metrics['mean_hold_time']:.2f} s")
        with col2:
            st.metric(label="Flight Time", value=f"{st.session_state.metrics['mean_flight_time']:.2f} s")
        with col3:
            st.metric(label="Variability", value=f"{st.session_state.metrics['rhythm_variability']:.2f}")
        with col4:
            st.metric(label="Pauses (>0.5s)", value=f"{st.session_state.metrics['pause_count']}")
        with col5:
            st.metric(label="Total K/S", value=f"{st.session_state.metrics['typing_speed']:.1f}")

        st.subheader("Calculated Indices")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Instability Score", value=f"{st.session_state.metrics['instability_score']:.2f}")
        with col2:
            st.metric(label="Interpretation", value=st.session_state.metrics['instability_level'])
        
        st.caption(f"Total Raw Keystroke events captured: {len(st.session_state.keystrokes)}")
