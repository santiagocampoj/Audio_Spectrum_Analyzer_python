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
import argparse



def record_audio(logger, duration=None):
    if duration is not None:
        logger.info(f"Recording setup for {duration} seconds.")
    else:
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
    audio_file_name = f"recording_{timestamp}.wav"

    logger.info("Saving recording...")
    path_wav = f"{timestamp}/{audio_file_name}"
    sf.write(path_wav, audio, RATE)

    logger.info(f"Recording saved to {path_wav}")

    return path_wav, duration, timestamp

def transcribe_audio(audio, logger):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio) as source:
        audio = recognizer.record(source)

    try:
        logger.info("Transcribing audio...")
        text = recognizer.recognize_google(audio)
        logger.info(f"Transcribed text: {text}")

    except sr.UnknownValueError:
        logger.warning("Could not understand audio")
    except sr.RequestError as e:
        logger.error(f"Could not request results from Google Speech Recognition service; {e}")

    return text

def plot_waveform(audio_file_path, duration, timestamp, logger):
    audio_data, _ = sf.read(audio_file_path)
    t = np.linspace(0, duration, len(audio_data))

    plt.figure(figsize=(10, 4))

    plt.plot(t, audio_data)
    plt.title("Waveform")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.savefig(f'{timestamp}/waveform_{timestamp}.png')
    logger.info("Waveform saved")


def arg_parser():
    parser = argparse.ArgumentParser(description="Audio stream")
    parser.add_argument("-d", "--duration", type=int, default=None, help="Duration of the recording in seconds")
    parser.add_argument("-f", "--file", type=str, default=None, help="Path to audio file")
    return parser.parse_args()


def main():
    args = arg_parser()
    logger = setup_logging()

    if args.duration:
        duration = args.duration
    else:
        duration = None

    logger.info(f"Constants: DURATION \t{duration}")

    logger.info(f"Constants: CHUNK \t\t{CHUNK}")
    logger.info(f"Constants: FORMAT \t\t{FORMAT}")
    logger.info(f"Constants: CHANNELS \t{CHANNELS}")
    logger.info(f"Constants: RATE \t\t{RATE}")


    if args.file:
        audio_file_path = args.file
        logger.info(f"Using audio file {audio_file_path}")

        if duration is None:
            audio_data, sr = sf.read(audio_file_path)
            duration = len(audio_data) / RATE

        logger.info(f"Duration: {duration}")

        if "/" in audio_file_path:
            timestamp = audio_file_path.split("/")[1].split("_")[1].split(".")[0]
            logger.info(f"Timestamp: {timestamp}")
        else:
            timestamp = audio_file_path.split("_")[1].split(".")[0]
            logger.info(f"Timestamp: {timestamp}")

        make_dir(timestamp)

        logger.info(f"Transcribing audio from {audio_file_path}...")
        text = transcribe_audio(audio_file_path, logger)

        plot_waveform(audio_file_path, duration, timestamp, logger)
        logger.info("Waveform plotted")


    else:
        # record and get audio file path
        audio_file_path, duration, timestamp = record_audio(logger, duration)
        make_dir(timestamp)

        # transcribe the audio from the file
        text = transcribe_audio(audio_file_path, logger)
        logger.info(f"Transcribed text: {text}")

        # Plot the waveform from the audio file
        plot_waveform(audio_file_path, duration, timestamp, logger)
        logger.info("Waveform plotted")


if __name__ == '__main__':
    main()