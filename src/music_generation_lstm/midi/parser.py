# load midi file(s) from given path and return them converted to music21 score
import os
import glob
from music21 import converter, stream
from config import RAW_MIDI_DIR

def parse_midi(dataset : str) -> list[stream.Score]:
    #
    #
    #

    path = os.path.join(RAW_MIDI_DIR, dataset)

    print(f"Started parsing at {path}")

    scores = []
    if os.path.isdir(path):
        print("Folder found")
        midi_files = glob.glob(os.path.join(path, "*.mid"))
        total = len(midi_files)
        for curr, file in enumerate(midi_files, start=1):
            print(f"[Parsing progress] {curr}/{total}", end="\r")
            try:
                scores.append(converter.parse(file))
            except Exception as e:
                print(e)
        print("")
    elif os.path.isfile(path):
        print("File found")
        if path[-4:] == ".mid":
            scores.append(converter.parse(path))
    else:
        raise Exception("Invalid path.")

    print(f"{len(scores)} midi files parsed")
    return scores


# Assert:
# empty, since no midi files in dir
# empty, since file is no midi
# single midi file
# multiple midi files
# invalid path

    parse_midi("/workspaces/music-generator/data/midi/raw/")

    parse_midi("/workspaces/music-generator/data/midi/raw/.gitkeep")

    parse_midi("/workspaces/music-generator/data/midi/raw/kpop_1_dataset/0.mid")

    parse_midi("/workspaces/music-generator/data/midi/raw/kpop_110_dataset")

    parse_midi("")
