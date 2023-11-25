import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from constants import *
import numpy as np
from scipy.fft import fft, fftfreq
import os
import datetime
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from utils import *
from logger_info import *

def record_audio(logger):
    duration = int(input("Enter the duration of the recording in seconds: "))

    logger.info(f"Recording setup for {duration} seconds.")
    
    logger.info("Recording audio...")
    audio = sd.rec(int(duration * RATE), samplerate=RATE, channels=CHANNELS)
    
    with tqdm(total=duration, desc="Recording", unit="sec", leave=False) as pbar:
        for _ in range(duration):
            time.sleep(1)
            pbar.update(1)
    
    sd.wait()

    logger.info("Recording finished")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    audio_file_path = f"recording_{timestamp}.wav"

    logger.info("Saving recording...")
    sf.write(audio_file_path, audio, RATE)
    logger.info(f"Recording saved to {audio_file_path}")

    return audio_file_path, duration, timestamp

def transcribe_audio(audio, logger):
    recognizer = sr.Recognizer()

    logger.info("Transcribing audio...")
    with sr.AudioFile(audio) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        logger.info(f"Transcribed text: {text}")
    except sr.UnknownValueError:
        logger.warning("Could not understand audio")
    except sr.RequestError as e:
        logger.error(f"Could not request results from Google Speech Recognition service; {e}")

    return text

def plot_waveform(audio_file_path, duration, timestamp):
    audio_data, _ = sf.read(audio_file_path)
    t = np.linspace(0, duration, len(audio_data))

    plt.figure(figsize=(10, 4))

    plt.plot(t, audio_data)
    plt.title("Waveform")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.savefig(f'waveform_{timestamp}.png')
    logger.info("Waveform saved")


def arg_parser():
    pass

def main():
    # Setup logging
    logger = setup_logging()

    logger.info(f"""
            Constants:

            CHUNK \t{CHUNK}
            FORMAT \t{FORMAT}
            CHANNELS \t{CHANNELS}
            RATE \t{RATE}
            
            """)

    # Record and get audio file path
    audio_file_path, duration, timestamp = record_audio(logger)
    make_dir(timestamp)

    # Transcribe the audio from the file
    logger.info("Transcribing audio...")
    text = transcribe_audio(audio_file_path, logger)
    logger.info(f"Transcribed text: {text}")

    # Plot the waveform from the audio file
    logger.info("Plotting waveform...")
    plot_waveform(audio_file_path, duration, timestamp)
    logger.info("Waveform plotted")

    logger.info("Program finished")

if __name__ == '__main__':
    main()