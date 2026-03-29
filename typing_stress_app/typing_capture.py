import time
from pynput import keyboard

class TypingCapture:
    def __init__(self, duration=10):
        """
        Initializes the typing capture module.
        :param duration: Time in seconds to capture typing.
        """
        self.duration = duration
        self.keystrokes = []
        self._pressed_keys = {}
        self.is_recording = False
        self.listener = None

    def on_press(self, key):
        """Called when a key is pressed."""
        if not self.is_recording:
            return
        
        # We use str(key) to represent the key (handles both characters and special keys)
        key_name = str(key)
        
        # Record the time the key was pressed.
        # Only record if it wasn't already pressed (to ignore key repeat events while holding).
        if key_name not in self._pressed_keys:
            self._pressed_keys[key_name] = time.time()

    def on_release(self, key):
        """Called when a key is released."""
        if not self.is_recording:
            return False # Stop listener
            
        key_name = str(key)
        release_time = time.time()
        
        # If we have a record of when this key was pressed, we can compute the hold duration
        if key_name in self._pressed_keys:
            press_time = self._pressed_keys.pop(key_name)
            self.keystrokes.append({
                'key': key_name,
                'press_time': press_time,
                'release_time': release_time
            })
            
    def start_capture(self):
        """
        Starts recording keystrokes for the specified duration.
        :return: A list of dictionaries containing key, press_time, and release_time.
        """
        print(f"Starting to capture keystrokes for {self.duration} seconds...")
        self.keystrokes = []
        self._pressed_keys = {}
        self.is_recording = True
        
        # Start the keyboard listener in a non-blocking thread
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()
        
        # Wait for the specified duration while capturing in the background
        start_time = time.time()
        while time.time() - start_time < self.duration:
            time.sleep(0.1) # Small sleep to prevent high CPU usage
            
        # Stop recording
        self.is_recording = False
        self.listener.stop()
        print("Finished capturing keystrokes.")
        
        return self.keystrokes
