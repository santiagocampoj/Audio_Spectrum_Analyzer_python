import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from constants import *
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

    # Trim the audio data
    audio = trim_audio_start(audio, RATE, TRIM_DURATION, logger)

    # Save the audio data
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    audio_file_name = f"recording_{timestamp}.wav"

    logger.info("Saving recording...")
    make_dir(timestamp)
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


def arg_parser():
    parser = argparse.ArgumentParser(description="Audio stream")
    parser.add_argument("-d", "--duration", type=int, default=None, help="Duration of the recording in seconds")
    parser.add_argument("-f", "--file", type=str, default=None, help="Path to audio file")
    return parser.parse_args()


def main():
    args = arg_parser()
    logger = setup_logging()

    logger.info(f"Constants: CHUNK \t\t{CHUNK} (samples per frame)")
    logger.info(f"Constants: FORMAT \t\t{FORMAT} (bytes per sample)")
    logger.info(f"Constants: CHANNELS \t{CHANNELS} (single channel for microphone)")
    logger.info(f"Constants: RATE \t\t{RATE} (samples per second)")


    if args.file:
        audio_file_path = args.file
        logger.info(f"Using audio file {audio_file_path}")

        audio_data, sr = sf.read(audio_file_path)
        duration = len(audio_data) / sr

        logger.info(f"Duration: {duration}s")
        logger.info(f"Sampling rate: {sr} Hz")

        if "/" in audio_file_path:
            timestamp = audio_file_path.split("/")[1].split("_")[1].split(".")[0]
        else:
            timestamp = audio_file_path.split("_")[1].split(".")[0]

        make_dir(timestamp)

        logger.info(f"Transcribing audio {audio_file_path}...")
        text = transcribe_audio(audio_file_path, logger)

        plot_waveform(audio_file_path, duration, timestamp, logger)
        logger.info("Waveform plotted")

        plot_spectrum(audio_file_path, duration, timestamp, logger, sr)
        logger.info("Frequency spectrum plotted")

        plot_spectrogram(audio_file_path, duration, timestamp, logger, sr)
        logger.info("Spectrogram plotted")

        plot_mel_spectrogram(audio_file_path, timestamp, logger, RATE)
        logger.info("Mel Spectrogram plotted")

        plot_mfcc(audio_file_path, timestamp, logger, RATE)
        logger.info("MFCC plotted")


    else:
        if args.duration:
            duration = args.duration
        else:
            duration = None

        # record and get audio file path
        audio_file_path, duration, timestamp = record_audio(logger, duration)
        make_dir(timestamp)

        # transcribe the audio from the file
        text = transcribe_audio(audio_file_path, logger)
        logger.info(f"Transcribed text: {text}")

        # Plot the waveform from the audio file
        plot_waveform(audio_file_path, duration, timestamp, logger)
        logger.info("Waveform plotted")

        # Plot the frequency spectrum from the audio file
        plot_spectrum(audio_file_path, duration, timestamp, logger, RATE)
        logger.info("Frequency spectrum plotted")

        plot_spectrogram(audio_file_path, duration, timestamp, logger, RATE)
        logger.info("Spectrogram plotted")

        plot_mel_spectrogram(audio_file_path, timestamp, logger, RATE)
        logger.info("Mel Spectrogram plotted")

        plot_mfcc(audio_file_path, timestamp, logger, RATE)
        logger.info("MFCC plotted")


if __name__ == '__main__':
    main()