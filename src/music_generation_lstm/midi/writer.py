# save music21 score as midi in the result folder

import os

from music21.stream import Stream
from config import RESULTS_MIDI_DIR

def write_midi(result_id : str, stream : Stream):
    print(f"Started saving {result_id}...", end="\r")
    write_dir = os.path.join(RESULTS_MIDI_DIR, result_id)
    os.makedirs(write_dir, exist_ok=False)
    stream.write("midi", fp=os.path.join(write_dir, f"{result_id}.midi"))
    print(f"Finished saving {result_id}.")
