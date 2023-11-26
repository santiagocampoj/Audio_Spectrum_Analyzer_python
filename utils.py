import os
import numpy as np
from scipy.fft import fft, fftfreq
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
import librosa

def make_dir(timestamp):
    """Make directory if it does not exist."""
    if not os.path.exists(timestamp):
        os.makedirs(timestamp)

def trim_audio_start(audio_data, sr, trim_duration, logger):
    trim_samples = int(sr * trim_duration)
    if len(audio_data) > trim_samples:
        logger.info(f"Trimmed {trim_duration}s from start of audio")
        return audio_data[trim_samples:]
    else:
        logger.info("Audio not trimmed. Audio length is less than trim duration")
        return audio_data


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

    # Calculate amplitude
    amplitude = np.abs(audio_data_fft)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, amplitude)
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.xlim([0, sr / 2])

    # Set y-axis limits dynamically based on the maximum amplitude
    ymax = np.percentile(amplitude, 99) #(amplitude) * 1.1  # A bit higher than the max amplitude
    plt.ylim([0, ymax])

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
    vmin = Sxx_log.max() - 90  # tweak this for gradient
    vmax = Sxx_log.max()

    # Plot the spectrogram
    pcm = plt.pcolormesh(t, f, Sxx_log, shading='gouraud', cmap='inferno', vmin=vmin, vmax=vmax)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')

    # Set colorbar ticks
    tick_interval = 10  # Define the interval of ticks here
    ticks = np.arange(vmin, vmax + tick_interval, tick_interval)

    # Create the colorbar with specified ticks
    cbar = plt.colorbar(pcm, label='Intensity [dB]', ticks=ticks)

    plt.savefig(f'{timestamp}/spectrogram_{timestamp}.png')  # Save the plot
    logger.info("Spectrogram saved")

def plot_mel_spectrogram(audio_file_path, timestamp, logger, sr, n_mels=256, n_fft=2048, hop_length=512):
    # Load audio data
    audio_data, _ = sf.read(audio_file_path)

    # Compute the Mel spectrogram
    S = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plotting
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=sr/2)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()

    # Save the Mel Spectrogram plot
    plt.savefig(f'{timestamp}/mel_spectrogram_{timestamp}.png')
    logger.info("Mel Spectrogram plot saved")


def plot_mfcc(audio_file_path, timestamp, logger, sr, n_mfcc=18):
    audio_data, _ = sf.read(audio_file_path)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)

    # Plotting
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()

    # Save the MFCC plot
    plt.savefig(f'{timestamp}/mfcc_{timestamp}.png')
    logger.info("MFCC plot saved")