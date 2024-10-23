import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import messagebox
from scipy.fft import fft
import wavio
import sounddevice as sd
from pydub import AudioSegment


class RecordGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Recorder")
        self.geometry("300x200")
        
        self.recording = False
        self.audio_file = "recording.wav"

        self.start_button = tk.Button(self, text="Start Recording", command=self.start_recording)
        self.start_button.pack(pady=20)

        self.stop_button = tk.Button(self, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(pady=20)

        self.audio_data = []
        self.stream = None

    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.audio_data = []
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # Start recording
            self.stream = sd.InputStream(callback=self.audio_callback)
            self.stream.start()

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
            # Stop recording
            if self.stream:
                self.stream.stop()
                self.stream.close()
            
            # Save the recording
            if self.audio_data:
                audio_np = np.concatenate(self.audio_data)
                wavio.write(self.audio_file, audio_np, 44100, sampwidth=2)
                messagebox.showinfo("Info", f"Recording saved as {self.audio_file}")
            self.destroy()

    def audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_data.append(indata.copy())

# Run the recording GUI
record_gui = RecordGUI()
record_gui.mainloop()
# Define frequency spectrum calculation
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
    
    return freq_array, freq_magnitude

NOTE_FREQUENCIES = {
    "E2": 82.41, "F2": 87.31, "F#2": 92.50, "G2": 98.00, "G#2": 103.83, "A2": 110.00, "A#2": 116.54, "B2": 123.47,
    "C3": 130.81, "C#3": 138.59, "D3": 146.83, "D#3": 155.56, "E3": 164.81, "F3": 174.61, "F#3": 185.00, "G3": 196.00,
    "G#3": 207.65, "A3": 220.00, "A#3": 233.08, "B3": 246.94, "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13,
    "E4": 329.63, "F4": 349.23, "F#4": 369.99, "G4": 392.00, "G#4": 415.30, "A4": 440.00, "A#4": 466.16, "B4": 493.88,
    "C5": 523.25, "C#5": 554.37, "D5": 587.33, "D#5": 622.25, "E5": 659.25, "F5": 698.46, "F#5": 739.99, "G5": 783.99,
    "G#5": 830.61, "A5": 880.00, "A#5": 932.33, "B5": 987.77, "C6": 1046.50, "C#6": 1108.73, "D6": 1174.66, "D#6": 1244.51,
    "E6": 1318.51, "F6": 1396.91, "F#6": 1479.98, "G6": 1567.98, "G#6": 1661.22, "A6": 1760.00, "A#6": 1864.66, "B6": 1975.53,
    "C7": 2093.00, "C#7": 2217.46, "D7": 2349.32, "D#7": 2489.02, "E7": 2637.02, "F7": 2793.83, "F#7": 2959.96, "G7": 3135.96,
    "G#7": 3322.44, "A7": 3520.00, "A#7": 3729.31, "B7": 3951.07, "C8": 4186.01
}

def closest_note_from_frequency(frequency):
    closest_note = min(NOTE_FREQUENCIES, key=lambda note: abs(NOTE_FREQUENCIES[note] - frequency))
    diff = frequency - NOTE_FREQUENCIES[closest_note]
    return closest_note, diff

def classify_note_attempt_1(peak_freq):
    closest_note, _ = closest_note_from_frequency(peak_freq)
    return closest_note

def classify_note_attempt_2(peak_freq, threshold=5):
    closest_note, diff = closest_note_from_frequency(peak_freq)
    if abs(diff) > threshold:
        return "Unknown"
    return closest_note

def classify_note_attempt_3(frequencies, magnitudes, threshold=5):
    primary_peak = frequencies[np.argmax(magnitudes)]
    harmonic_candidates = []
    for harmonic_multiplier in range(1, 6):
        harmonic_freq = primary_peak * harmonic_multiplier
        closest_note, diff = closest_note_from_frequency(harmonic_freq)
        if abs(diff) <= threshold:
            harmonic_candidates.append((closest_note, diff))
    if harmonic_candidates:
        return min(harmonic_candidates, key=lambda x: abs(x[1]))[0]
    else:
        return "Unknown"

def predict_notes(song, starts, actual_notes=None, plot_fft_indices=[]):
    predicted_notes = []
    
    NOTE_DURATION_MS = 500
    
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
    SEGMENT_MS = 50
    VOLUME_THRESHOLD = -20
    EDGE_THRESHOLD = 5
    MIN_MS_BETWEEN = 100

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

GUITAR_STRINGS = {
    "E": 82.41,
    "A": 110.00,
    "D": 146.83,
    "G": 196.00,
    "B": 246.94,
    "e": 329.63
}

def guitar_fretboard_notes():
    notes = {}
    for string, open_freq in GUITAR_STRINGS.items():
        for fret in range(25):
            freq = open_freq * (2 ** (fret / 12.0))
            note, _ = closest_note_from_frequency(freq)
            notes[(string, fret)] = note
    return notes

class GuitarFretboardGUI(tk.Tk):
    def __init__(self, predicted_notes):
        super().__init__()
        self.title("Guitar Fretboard")
        self.geometry("800x600")
        
        self.predicted_notes = predicted_notes
        self.notes_on_fretboard = guitar_fretboard_notes()
        self.notes_to_display = set(predicted_notes)  # Set of predicted notes to be displayed
        
        self.canvas = tk.Canvas(self, bg='white', width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.draw_fretboard()
        self.display_predicted_notes()

    def draw_fretboard(self):
        self.canvas.create_rectangle(50, 50, 750, 550, outline="black", fill="lightgray")
        for string_idx, string_name in enumerate(GUITAR_STRINGS):
            y = 100 + string_idx * 80
            self.canvas.create_line(50, y, 750, y, fill="black")
        for fret in range(25):
            x = 50 + fret * 25
            self.canvas.create_line(x, 50, x, 550, fill="black")

    def display_predicted_notes(self):
        for (string, fret), note in self.notes_on_fretboard.items():
            if note in self.notes_to_display:
                x = 50 + fret * 25
                y = 100 + list(GUITAR_STRINGS.keys()).index(string) * 80
                self.canvas.create_text(x, y, text=note, fill="red", font=("Arial", 8))

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
    app = GuitarFretboardGUI(predicted_notes)
    app.mainloop()

# Run the recording GUI first
record_gui = RecordGUI()
record_gui.mainloop()

# After recording, process the recorded audio
main("recording.wav", plot_starts=True)
