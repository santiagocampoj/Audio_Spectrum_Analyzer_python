import pyaudio
from logger_info import setup_logging
logger = setup_logging()

# Constants
CHUNK = 1024 * 4  # Samples per frame = 4096
FORMAT = pyaudio.paInt16  # Audio format (bytes per sample)
CHANNELS = 1  # Single channel for microphone
RATE = 44100  # Samples per second

logger.info(f"""
            Constants:

            CHUNK \t{CHUNK}
            FORMAT \t{FORMAT}
            CHANNELS \t{CHANNELS}
            RATE \t{RATE}
            
            """)