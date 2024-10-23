import tkinter as tk
import sounddevice as sd
import wavio
from tkinter import messagebox
import numpy as np

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
            self.stream: sd.InputStream = sd.InputStream(callback=self.audio_callback)
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