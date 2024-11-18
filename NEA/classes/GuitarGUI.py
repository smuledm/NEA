import tkinter as tk
from .Constants import *


class GuitarFretboardGUI(tk.Tk):
    def __init__(self, predicted_notes):
        super().__init__()
        self.title("Guitar Fretboard")
        self.geometry("800x600")
        
        self.predicted_notes = predicted_notes
        self.notes_on_fretboard = self.guitar_fretboard_notes()
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

    def guitar_fretboard_notes():
        notes = {}
        for string, open_freq in GUITAR_STRINGS.items():
            for fret in range(25):
                freq = open_freq * (2 ** (fret / 12.0))
                note, _ = closest_note_from_frequency(freq)
                notes[(string, fret)] = note
        return notes