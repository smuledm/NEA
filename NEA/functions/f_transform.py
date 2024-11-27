import numpy as np
from scipy.fft import fft
from pydub import AudioSegment
import matplotlib.pyplot as plt

def frequency_spectrum(sample, max_frequency=4500):
          
    samples = np.array(sample.get_array_of_samples())
    sample_rate = sample.frame_rate
    n = len(samples)
    
    freq_magnitude = fft(samples)
    freq_array = np.fft.fftfreq(n, 1 / sample_rate)[:n // 2]
    freq_magnitude = np.abs(freq_magnitude[:n // 2])
    
    if max_frequency:
        max_index = int(max_frequency * n / sample_rate) + 1
        freq_array = freq_array[:max_index]
        freq_magnitude = freq_magnitude[:max_index]
    
    freq_magnitude = freq_magnitude / np.sum(freq_magnitude)
    
    audio_file = "C:/Users/samue/Documents/github/NEA/recording.wav"
    audio = AudioSegment.from_file(audio_file)

    # Slice 500ms of the audio (adjust start and end as needed)
    audio_sample = audio[:500]

    # Get the frequency spectrum


    # Plots the spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(freq_array, freq_magnitude, color='blue')
    plt.title("Frequency Spectrum of Audio Sample", fontsize=14)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Magnitude |X(freq)|", fontsize=12)
    plt.grid()
    plt.show()
    
    
    return freq_array, freq_magnitude



