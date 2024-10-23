import pydub
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import messagebox
from scipy.fft import fft
import wavio
import os
import sounddevice as sd
from pydub.utils import get_array_type
import array
from pyaudio import get_sa








sample = "recording.wav"


def frequency_spectrum(sample, max_frequency=800):
    """
    Derive frequency spectrum of a pydub.AudioSample
    Returns an array of frequencies and an array of how prevalent that frequency is in the sample
    """
    bit_depth = sample. * 8
    array_type = get_array_type(bit_depth)
    raw_audio_data = array.array(array_type, sample._data)
    n = len(raw_audio_data)
 
    freq_array = np.arange(n) * (float(sample.frame_rate) / n)  # two sides frequency range
    freq_array = freq_array[:(n // 2)]  # one side frequency range
    raw_audio_data = raw_audio_data - np.average(raw_audio_data)  # zero-centering
    
    freq_magnitude = fft(raw_audio_data) # fft computing and normalization
    freq_magnitude = freq_magnitude[:(n // 2)] # one side
    if max_frequency:
        max_index = int(max_frequency * n / sample.frame_rate) + 1
        freq_array = freq_array[:max_index]
        freq_magnitude = freq_magnitude[:max_index]
    freq_magnitude = freq_magnitude / np.sum(freq_magnitude)
    plt.plot(freq_array, freq_magnitude, 'b')
    return freq_array, freq_magnitude

frequency_spectrum(sample)