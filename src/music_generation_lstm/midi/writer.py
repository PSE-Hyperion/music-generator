# save music21 score as midi in the result folder

import os

from music21.stream import Stream
from config import RESULT_MIDI_DIR

def write_midi(id : str, stream : Stream):
    print(f"Started saving {id}...", end="\r")
    write_dir = os.path.join(RESULT_MIDI_DIR, id)
    os.makedirs(write_dir, exist_ok=False)
    stream.write("midi", fp=os.path.join(write_dir, f"{id}.mid"))
    print(f"Finished saving {id}.")
