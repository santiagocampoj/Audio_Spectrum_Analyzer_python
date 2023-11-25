import os
import numpy as np
from scipy.fft import fft, fftfreq
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal

def make_dir(timestamp):
    """Make directory if it does not exist."""
    if not os.path.exists(timestamp):
        os.makedirs(timestamp)

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

def plot_spectrum(audio_file_path, duration, timestamp, logger, sr):
    audio_data, _ = sf.read(audio_file_path)
    audio_data_fft = fft(audio_data)
    freqs = fftfreq(len(audio_data), 1 / sr)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, np.abs(audio_data_fft))
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.xlim([0, sr / 2])
    plt.legend()

    plt.savefig(f'{timestamp}/spectrum_{timestamp}.png')
    logger.info("Frequency spectrum saved")

def plot_spectrogram(audio_file_path, duration, timestamp, logger, sr):
    audio_data, _ = sf.read(audio_file_path)  # Read the audio data from the file

    plt.figure(figsize=(10, 4))
    
    f, t, Sxx = signal.spectrogram(audio_data, sr)  # Calculate the spectrogram

    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='inferno')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    plt.colorbar(label='Intensity [dB]')

    plt.savefig(f'{timestamp}/spectrogram_{timestamp}.png')  # Save the plot
    logger.info("Spectrogram saved")