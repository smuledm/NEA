from classes.Constants import *
import numpy as np

def closest_note_from_frequency(frequency: int) -> tuple[int]:
    '''

    longer description

    Inputs
    ------

    Returns
    -------
    
    '''
 
    closest_note = min(NOTE_FREQUENCIES, key=lambda note: abs(NOTE_FREQUENCIES[note] - frequency))
    return closest_note












