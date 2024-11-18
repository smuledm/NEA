from classes.Constants import *
import numpy as np

def closest_note_from_frequency(frequency: int) -> tuple[int]:
    '''
    short description

    longer description

    Inputs
    ------

    Returns
    -------
    
    '''
    closest_note = min(NOTE_FREQUENCIES, key=lambda note: abs(NOTE_FREQUENCIES[note] - frequency))
    diff = frequency - NOTE_FREQUENCIES[closest_note]
    return closest_note, diff

def classify_note_attempt_1(peak_freq: int) -> int:
    closest_note, _ = closest_note_from_frequency(peak_freq)
    return closest_note

def classify_note_attempt_2(peak_freq, threshold=5) -> int:
    closest_note, diff = closest_note_from_frequency(peak_freq)
    if abs(diff) > threshold:
        return "Unknown"
    return closest_note

def classify_note_attempt_3(frequencies, magnitudes, threshold=5) -> int:
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
    

def determine_note(peak_freq: int, ) -> int:
    attempt1 = classify_note_attempt_1(peak_freq)
    attempt2 = classify_note_attempt_2(peak_freq, threshold=5)
    attempt3 = classify_note_attempt_3(threshold=5)

    











