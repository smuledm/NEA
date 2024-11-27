from classes.Constants import *
import numpy as np

def closest_note_from_frequency(frequency: int) -> tuple[int]:
    '''

    longer description
    Gives the closest note from the frequency aquired by the FFT

    Inputs
    Takes the frequencies of the notes played

    Returns
    A tuple of the frequencies
    
    '''
 
    closest_note = min(NOTE_FREQUENCIES, key=lambda note: abs(NOTE_FREQUENCIES[note] - frequency))
    return closest_note












