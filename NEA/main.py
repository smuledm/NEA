import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from pydub import AudioSegment
from classes import *
from functions import *





def predict_notes(song, starts, actual_notes=None, plot_fft_indices=[]):
    predicted_notes = []
    
    for i, start_ms in enumerate(starts):
        note_segment = song[start_ms:start_ms + NOTE_DURATION_MS]
        samples = np.array(note_segment.get_array_of_samples())
        sample_rate = note_segment.frame_rate
        
        freqs, magnitudes = frequency_spectrum(note_segment, sample_rate)
        peak_freq = freqs[np.argmax(magnitudes)]
        
        predicted_note = classify_note_attempt_1(peak_freq)
        predicted_notes.append(predicted_note)
        
        print(f"Start time: {start_ms} ms")
        print(f"Predicted Frequency: {peak_freq:.2f} Hz")
        print(f"Predicted Note: {predicted_note}")
        
        if i in plot_fft_indices:
            plt.figure(figsize=(10, 4))
            plt.plot(freqs, magnitudes)
            plt.title(f"FFT of Note at {start_ms} ms (Predicted as {predicted_note})")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.show()
    
    return predicted_notes

def calculate_distance(predicted_notes, actual_notes):
    n = len(predicted_notes)
    m = len(actual_notes)
    distance_matrix = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        distance_matrix[i][0] = i
    for j in range(m + 1):
        distance_matrix[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if predicted_notes[i - 1] == actual_notes[j - 1]:
                cost = 0
            else:
                cost = 1
            distance_matrix[i][j] = min(
                distance_matrix[i - 1][j] + 1,
                distance_matrix[i][j - 1] + 1,
                distance_matrix[i - 1][j - 1] + cost
            )
    return distance_matrix[n][m]

def predict_note_starts(song, plot, actual_starts):


    song = song.high_pass_filter(80)
    volume = [segment.dBFS for segment in song[::SEGMENT_MS]]

    predicted_starts = []
    for i in range(1, len(volume)):
        if volume[i] > VOLUME_THRESHOLD and volume[i] - volume[i - 1] > EDGE_THRESHOLD: # VOLUME CHECK
            ms = i * SEGMENT_MS
            if len(predicted_starts) == 0 or ms - predicted_starts[-1] >= MIN_MS_BETWEEN:
                predicted_starts.append(ms)

    if len(actual_starts) > 0:
        print("Approximate actual note start times ({})".format(len(actual_starts)))
        print(" ".join(["{:5.2f}".format(s) for s in actual_starts]))
        print("Predicted note start times ({})".format(len(predicted_starts)))
        print(" ".join(["{:5.2f}".format(ms / 1000) for ms in predicted_starts]))

    if plot:
        x_axis = np.arange(len(volume)) * (SEGMENT_MS / 1000)
        plt.figure(figsize=(12, 6))
        plt.plot(x_axis, volume, label='Volume', color='blue')
        
        for s in actual_starts:
            plt.axvline(x=s, color="red", linewidth=1.0, linestyle="-", label='Actual Start Time' if s == actual_starts[0] else "")
        
        for ms in predicted_starts:
            plt.axvline(x=(ms / 1000), color="green", linewidth=1.0, linestyle="--", label='Predicted Start Time' if ms == predicted_starts[0] else "")
        
        plt.xlabel("Time (s)")
        plt.ylabel("Volume (dBFS)")
        plt.title("Volume and Note Start Times")
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return predicted_starts

def main(file, note_file=None, note_starts_file=None, plot_starts=False, plot_fft_indices=[]):
    actual_starts = []
    if note_starts_file:
        with open(note_starts_file) as f:
            for line in f:
                actual_starts.append(float(line.strip()))
    actual_notes = []
    if note_file:
        with open(note_file) as f:
            for line in f:
                actual_notes.append(line.strip())
    
    song = AudioSegment.from_file(file)
    song = song.high_pass_filter(80)
    starts = predict_note_starts(song, plot_starts, actual_starts)
    predicted_notes = predict_notes(song, starts, actual_notes, plot_fft_indices)
    
    print("")
    if actual_notes:
        print("Actual Notes")
        print(actual_notes)
    print("Predicted Notes")
    print(predicted_notes)
    
    if actual_notes:
        lev_distance = calculate_distance(predicted_notes, actual_notes)
        print("Levenshtein distance: {}/{}".format(lev_distance, len(actual_notes)))
    
    # Show the GUI with predicted notes
   # app = GuitarFretboardGUI(predicted_notes)
   # app.mainloop()

record_gui = RecordGUI()
record_gui.mainloop()
# After recording, process the recorded audio
main("recording.wav", plot_starts=True)