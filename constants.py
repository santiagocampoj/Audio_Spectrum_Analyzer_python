import pyaudio

# Constants
CHUNK = 1024 * 4  # Samples per frame = 4096
FORMAT = pyaudio.paInt16  # Audio format (bytes per sample)
CHANNELS = 1  # Single channel for microphone
RATE = 44100  # Samples per second
TRIM_DURATION = 0.5  # Trim seconds from start of audio