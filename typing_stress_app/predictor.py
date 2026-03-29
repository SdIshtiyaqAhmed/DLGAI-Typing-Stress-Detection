import os
import sys
import joblib
import numpy as np
import warnings
import sys
import importlib.metadata

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

# Suppress common warnings from scikit-learn and tensorflow for cleaner terminal output
warnings.filterwarnings('ignore')

class Predictor:
    def __init__(self, model_dir='.'):
        """
        Initializes the predictor, attempting to load models from the given directory.
        :param model_dir: Path to look for the model files.
        """
        self.model_dir = model_dir
        self.scaler = None
        self.keras_model = None
        self.rf_model = None
        self._load_models()

    def _find_model_path(self, filename):
        """
        Helper method to locate model files robustly.
        Searches the current directory, parent directory, and provided model_dir.
        """
        search_paths = [self.model_dir, '.', '..']
        for path in search_paths:
            full_path = os.path.join(path, filename)
            if os.path.exists(full_path):
                return full_path
        return None

    def _load_models(self):
        """Loads the scaler, Keras deep learning model, and Random Forest fallback model safely."""
        
        # 1. Load Scaler
        scaler_path = self._find_model_path('scaler.pkl')
        if scaler_path:
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"Successfully loaded {scaler_path}")
                sys.stdout.flush()
            except Exception as e:
                print(f"Error loading scaler: {e}")
                sys.stdout.flush()
        else:
            print("Warning: scaler.pkl not found. Features will NOT be scaled before prediction.")
            sys.stdout.flush()

        # 2. Load Keras Deep Learning Model (Primary)
        keras_path = self._find_model_path('stress_detection_model.keras')
        if keras_path:
            try:
                # Local import to prevent loading TF unnecessarily if the model doesn't exist
                from tensorflow.keras.models import load_model 
                import tensorflow as tf
                
                # Definitive fix: Monkey-patch layers to ignore incompatible arguments
                from tensorflow.keras.layers import Dense, Dropout
                
                def make_robust(cls, *args_to_pop):
                    original_init = cls.__init__
                    def robust_init(self, *args, **kwargs):
                        for arg in args_to_pop:
                            kwargs.pop(arg, None)
                        return original_init(self, *args, **kwargs)
                    cls.__init__ = robust_init
                
                # Apply patches to Dense and Dropout
                make_robust(Dense, 'quantization_config')
                make_robust(Dropout, 'quantization_config')

                # Create a robust set of custom objects to absorb common version-mismatch errors
                class DummyObject:
                    def __init__(self, *args, **kwargs): pass
                    def __call__(self, *args, **kwargs): return 0.01
                    def get_config(self): return {}
                    @classmethod
                    def from_config(cls, config): return cls(**config)

                custom_objects = {
                    'schedules_distributions': DummyObject,
                    'quantization_config': DummyObject,
                    'safe_mode': DummyObject
                }
                
                # Attempt 1: Standard load
                try:
                    self.keras_model = load_model(keras_path, compile=False)
                except Exception as e:
                    print(f"Standard loading failed, trying robust direct: {e}")
                    # Attempt 2: Load with direct custom_objects
                    try:
                        self.keras_model = load_model(keras_path, compile=False, custom_objects=custom_objects)
                    except Exception as e2:
                        print(f"Robust direct failed, trying scope: {e2}")
                        # Attempt 3: Load within custom object scope
                        with tf.keras.utils.custom_object_scope(custom_objects):
                            try:
                                self.keras_model = load_model(keras_path, compile=False)
                            except Exception as e3:
                                print(f"Scope failed, trying keras.saving: {e3}")
                                # Attempt 4: Try lower-level keras.saving if available
                                try:
                                    import keras
                                    self.keras_model = keras.saving.load_model(keras_path, compile=False)
                                except Exception:
                                    # Final failure
                                    raise e3
                        
                print(f"Successfully loaded {keras_path}")
                sys.stdout.flush()
            except Exception as e:
                import traceback
                print(f"Error loading Keras model: {e}")
                traceback.print_exc()
                sys.stdout.flush()
        else:
            print("Warning: stress_detection_model.keras not found.")
            sys.stdout.flush()

        # 3. Load Scikit-Learn Random Forest Model (Alternative/Fallback)
        rf_path = self._find_model_path('stress_rf_model.pkl')
        if rf_path:
            try:
                self.rf_model = joblib.load(rf_path)
                print(f"Successfully loaded {rf_path}")
                sys.stdout.flush()
            except Exception as e:
                print(f"Error loading Random Forest model: {e}")
                sys.stdout.flush()
        else:
            print("Warning: stress_rf_model.pkl not found.")
            sys.stdout.flush()

    def predict(self, feature_vector):
        """
        Normalizes the extracted features and makes a prediction using all available trained models.
        :param feature_vector: 1D numpy array of timing features.
        :return: Dictionary containing predictions from Random Forest and Neural Networks.
        """
        results = {
            'nn_class': None,
            'nn_conf': 0.0,
            'rf_class': None,
            'rf_conf': 0.0,
            'agreement': False
        }
        
        if feature_vector is None or len(feature_vector) == 0:
            return results

        # Most models expect a 2D array representation for a single sample: shape (1, num_features)
        X = np.array(feature_vector).reshape(1, -1)

        # Scale features using the loaded scaler
        if self.scaler:
            try:
                X = self.scaler.transform(X)
            except Exception as e:
                print(f"Error scaling features: {e}")

        # Evaluate Keras Model
        if self.keras_model:
            try:
                # Keras models can be picky about input layers dimensions.
                # If the model was trained on different feature counts than the scaler, it will crash here.
                prediction_prob = self.keras_model.predict(X, verbose=0)
                
                # Check model output shape (e.g. binary output vs softmax classification)
                if prediction_prob.shape[-1] == 1:
                    prob = float(prediction_prob[0][0])
                    class_idx = 1 if prob >= 0.5 else 0
                    confidence = prob if class_idx == 1 else 1 - prob
                else:
                    class_idx = np.argmax(prediction_prob[0])
                    confidence = float(prediction_prob[0][class_idx])
                
                results['nn_class'] = str(class_idx)
                results['nn_conf'] = confidence
            except Exception as e:
                import traceback
                print(f"Error predicting with Keras model: {e}")
                print(traceback.format_exc())

        # Evaluate Random Forest model
        if self.rf_model:
            try:
                class_idx = self.rf_model.predict(X)[0]
                
                # If predict_proba is supported by the model type, fetch probability 
                if hasattr(self.rf_model, "predict_proba"):
                    probs = self.rf_model.predict_proba(X)[0]
                    confidence = float(np.max(probs))
                else:
                    confidence = 1.0 # Unknown probability

                results['rf_class'] = str(class_idx)
                results['rf_conf'] = confidence
            except Exception as e:
                print(f"Error predicting with Random Forest model: {e}")

        # Check for model agreement
        if results['nn_class'] is not None and results['rf_class'] is not None:
            results['agreement'] = (results['nn_class'] == results['rf_class'])
        else:
            # If one model failed, we cannot have agreement.
            results['agreement'] = None 

        return results
