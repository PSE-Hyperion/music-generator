# save music21 score as midi in the result folder

import os

from mido import MidiFile
from config import RESULTS_MIDI_DIR

def write_midi(result_id : str, midi : MidiFile):
    print(f"Started saving {result_id}...")
    write_dir = os.path.join(RESULTS_MIDI_DIR, result_id)
    os.makedirs(write_dir, exist_ok=False)
    midi.save(os.path.join(write_dir, f"{id}.midi"))
    print(f"Finished saving {result_id}.")
