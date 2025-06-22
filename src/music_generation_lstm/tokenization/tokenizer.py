# tokenizer class, that holds token to int map and can en- and decode token or integer lists

from music21 import converter, stream, note, chord

def quantize(value, resolution=1/8):
    return round(value / resolution) * resolution

class Tokenizer():
    def __init__(self):
        self.token_to_id = {}

    # builds the map
    def build(self, scores : list[stream.Score]):
        #
        #
        #

        print("Start building tokenizer...")

        total_tokens = []
        for score in scores:
            flat_score = score.flatten().notesAndRests.stream()

            tokens = []
            for el in flat_score:
                if el.isNote:
                    tokens.append(f"NOTE_ON_{el.pitch}_{quantize(el.quarterLength)}_{quantize(el.offset)}")
                elif el.isRest:
                    tokens.append(f"REST_{quantize(el.quarterLength)}_{quantize(el.offset)}")
                elif el.isChord:
                    pitches = '.'.join(str(p) for p in el.pitches)
                    tokens.append(f"CHORD_{pitches}_{quantize(el.quarterLength)}_{quantize(el.offset)}")
            total_tokens.extend(tokens)

        unique_tokens = list(set(total_tokens))
        print(len(unique_tokens))


    # uses map on token list
    def encode(self, scores : list[stream.Score]):
        print("Start encoding to tokens...")

        total_tokens = []
        for score in scores:
            flat_score = score.flatten().notesAndRests.stream()

            tokens = []
            for el in flat_score:
                if el.isNote:
                    tokens.append(f"NOTE_ON_{el.pitch}_{quantize(el.quarterLength)}_{quantize(el.offset)}")
                elif el.isRest:
                    tokens.append(f"REST_{quantize(el.quarterLength)}_{quantize(el.offset)}")
                elif el.isChord:
                    pitches = '.'.join(str(p) for p in el.pitches)
                    tokens.append(f"CHORD_{pitches}_{quantize(el.quarterLength)}_{quantize(el.offset)}")
            total_tokens.extend(tokens)

        unique_tokens = list(set(total_tokens))
        print(len(unique_tokens))

    # uses reversed map on integer list
    def decode(self):
        pass

    # save map to path (should be saved inside of the corresopnding model)
    def save(self):
        pass

    # load map from path (replaces build call)
    @classmethod
    def load(cls):
        pass



if False:
    from music21 import converter, stream, note, chord

    score = converter.parse("original_metheny.mid")

    flat = score.flatten().notesAndRests.stream()

    print("Start tokenize...")

    def quantize(value, resolution=1/8):
        return round(value / resolution) * resolution

    tokens = []
    for el in flat:
        offset = el.offset
        if el.isNote:
            tokens.append(f"NOTE_ON_{el.pitch}_{quantize(el.quarterLength, resolution=1/4)}_{quantize(el.offset, resolution=1/4)}")
        elif el.isRest:
            tokens.append(f"REST_{quantize(el.quarterLength, resolution=1/4)}_{quantize(el.offset, resolution=1/4)}")
        elif el.isChord:
            pitches = '.'.join(str(p) for p in el.pitches)
            tokens.append(f"CHORD_{pitches}_{quantize(el.quarterLength, resolution=1/4)}_{quantize(el.offset, resolution=1/4)}")

    unique_tokens = list(set(tokens))

    print(f"Amount of unique tokens: {len(unique_tokens)}")

    print("Start reverting...")

    s = stream.Stream()
    for token in tokens:
        parts = token.split("_")
        if token.startswith("NOTE_ON"):
            pitch, dur, offset = parts[2], float(parts[3]), float(parts[4])
            n = note.Note(pitch, quarterLength=dur)
            s.insert(offset, n)
        elif token.startswith("REST"):
            dur, offset = float(parts[1]), float(parts[2])
            r = note.Rest(quarterLength=dur)
            s.insert(offset, r)
        elif token.startswith("CHORD"):
            pitches, dur, offset = parts[1], float(parts[2]), float(parts[3])
            chord_obj = chord.Chord(pitches.split("."), quarterLength=dur)
            s.insert(offset, chord_obj)


    s.write("midi", fp="result_only_piano.mid")
    print("Saved result")
