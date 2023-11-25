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
    audio_data, _ = sf.read(audio_file_path)

    plt.figure(figsize=(10, 4))

    # Window size and overlap
    nperseg = 1024  # Window size
    noverlap = 768  # Overlap (75%)

    f, t, Sxx = signal.spectrogram(audio_data, sr, window='hann', nperseg=nperseg, noverlap=noverlap)

    Sxx_log = 10 * np.log10(Sxx)  # Convert to decibel scale

    # Adjust vmin and vmax for better contrast
    vmin = Sxx_log.max() - 110  # You might need to adjust this based on your data
    vmax = Sxx_log.max()

    # Use grayscale colormap for a look similar to Audacity
    plt.pcolormesh(t, f, Sxx_log, shading='gouraud', cmap='inferno', vmin=vmin, vmax=vmax)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    plt.colorbar(label='Intensity [dB]')

    plt.savefig(f'{timestamp}/spectrogram_{timestamp}.png')  # Save the plot
    logger.info("Spectrogram saved")