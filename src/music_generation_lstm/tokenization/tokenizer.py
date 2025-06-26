# tokenizer class, that holds token to int map and can en- and decode token or integer lists

from music21 import converter, stream, note, chord, instrument
from config import SEQUENCE_LENGTH, QUANTIZATION_PRECISION_DELTA_OFFSET, QUANTIZATION_PRECISION_DURATION
from fractions import Fraction

class EmbeddedTokenEvent():
    def __init__(self, type : str, pitch : str, duration : str, delta_offset : str, velocity : str, instrument : str):
        self.type = type
        self.pitch = pitch
        self.duration = duration
        self.delta_offset = delta_offset
        self.velocity = velocity
        self.instrument = instrument

def quantize(value, resolution : float = 1/8) -> float:
    return round(value / resolution) * resolution

def note_event(note : note.Note, instrument : instrument.Instrument, curr_offset : float, part_of_chord : bool = False, first_chord_note : bool = False) -> EmbeddedTokenEvent:
    label = "NOTE"
    if part_of_chord:
        label = "CHORD-NOTE"
        if first_chord_note:
            label = "CHORD-NOTE-START"             # was, wenn chords in unterschiedlichen parts gleichzeitig stattfinden

    return EmbeddedTokenEvent(
        f"TYPE_{label}",
        f"PITCH{note.pitch}",
        f"DURATION_{quantize(value=float(Fraction(note.quarterLength)), resolution=QUANTIZATION_PRECISION_DURATION)}",
        f"OFFSET_{quantize(value=curr_offset, resolution=QUANTIZATION_PRECISION_DELTA_OFFSET)}",
        f"VELOCITY_{note.volume.velocity}",
        f"INSTRUMENT_{instrument.instrumentName}"
    )

def rest_event(rest : note.Rest, curr_offset : float) -> EmbeddedTokenEvent:
    return EmbeddedTokenEvent(
        "REST",
        "NO_PITCH",
        f"DURATION_{quantize(value=float(Fraction(rest.quarterLength)), resolution=QUANTIZATION_PRECISION_DURATION)}",
        f"OFFSET_{quantize(value=curr_offset, resolution=QUANTIZATION_PRECISION_DELTA_OFFSET)}",
        "NO_VELOCITY",
        "NO_INSTRUMENT"
    )

class Tokenizer():
    def __init__(self, dataset_id : str):
        self.dataset_id = dataset_id
        self.type_map = {}
        self.pitch_map = {}
        self.duration_map = {}
        self.delta_offset_map = {}
        self.velocity_map = {}
        self.instrument_map = {}


    def extend_maps(self, embedded_token_events : list[EmbeddedTokenEvent]):
        #   extends the maps of this tokenizer instance
        #
        #

        types = set()
        pitches = set()
        durations = set()
        delta_offsets = set()
        velocities = set()
        instruments = set()

        # create sets, to extract unique tokens
        for embedded_token_event in embedded_token_events:
            type = embedded_token_event.type
            types.add(type)
            pitch = embedded_token_event.pitch
            pitches.add(pitch)
            duration = embedded_token_event.duration
            durations.add(duration)
            delta_offset = embedded_token_event.delta_offset
            delta_offsets.add(delta_offset)
            velocity = embedded_token_event.velocity
            velocities.add(velocity)
            instr = embedded_token_event.instrument
            instruments.add(instr)

        for type in types:
            if type not in self.type_map:
                self.type_map[type] = len(self.type_map)
        for type in types:
            if type not in self.pitch_map:
                self.pitch_map[type] = len(self.pitch_map)
        for type in types:
            if type not in self.duration_map:
                self.duration_map[type] = len(self.duration_map)
        for type in types:
            if type not in self.delta_offset_map:
                self.delta_offset_map[type] = len(self.delta_offset_map)
        for type in types:
            if type not in self.velocity_map:
                self.velocity_map[type] = len(self.velocity_map)
        for type in types:
            if type not in self.instrument_map:
                self.instrument_map[type] = len(self.instrument_map)

        print("Finished extending maps of unique tokens.")



    def save_maps(self):            # preliminary
        import json
        import os
        from config import TOKEN_MAPS_DIR

        print(f"Type map size: {len(self.type_map)}")
        print(f"Pitch map size: {len(self.pitch_map)}")
        print(f"Duration map size: {len(self.duration_map)}")
        print(f"Delta Offset map size: {len(self.delta_offset_map)}")
        print(f"Velocity map size: {len(self.velocity_map)}")
        print(f"Instrument map size: {len(self.instrument_map)}")

        total_unique_tokens = len(self.type_map) + len(self.pitch_map)+len(self.duration_map)+len(self.delta_offset_map)+len(self.velocity_map)+len(self.instrument_map)

        print(f"Total unique tokens: {total_unique_tokens}")

        self.sequence_length = SEQUENCE_LENGTH
        self.num_features_type = len(self.type_map)
        self.num_features_pitch = len(self.pitch_map)
        self.num_features_duration = len(self.duration_map)
        self.num_features_delta_offset = len(self.delta_offset_map)
        self.num_features_velocity = len(self.velocity_map)
        self.num_features_instrument = len(self.instrument_map)

        print("Start saving maps...")
        folder_path = os.path.join(TOKEN_MAPS_DIR, self.dataset_id)
        os.makedirs(folder_path, exist_ok=False)
        with open(os.path.join(folder_path, "type_map"), "w") as f:
            json.dump(self.type_map, f, indent=4)
        with open(os.path.join(folder_path, "pitch_map"), "w") as f:
            json.dump(self.pitch_map, f, indent=4)
        with open(os.path.join(folder_path, "duration_map"), "w") as f:
            json.dump(self.duration_map, f, indent=4)
        with open(os.path.join(folder_path, "delta_offset_map"), "w") as f:
            json.dump(self.delta_offset_map, f, indent=4)
        with open(os.path.join(folder_path, "velocity_map"), "w") as f:
            json.dump(self.velocity_map, f, indent=4)
        with open(os.path.join(folder_path, "instrument_map"), "w") as f:
            json.dump(self.instrument_map, f, indent=4)
        print("Finished saving maps.")


    def tokenize(self, score : stream.Score) -> list[EmbeddedTokenEvent]:   # return list of EmbeddedTokenEvents
        #   receives a score, that it will tokenize to EmbeddedTokenEvents
        #   EmbeddedTokenEvents is a group of tokens per event
        #   The score is turned into a list of embedded token events

        print("Start encoding to tokens...")

        embedded_token_events = []


        flat = score.flatten().notesAndRests            # .stream() if we want to have more control or access

        embedded_token_events = []
        curr_instrument = instrument.Instrument("Piano")

        prev_offset = 0.0

        note_counter = 0
        rest_counter = 0
        chord_counter = 0
        note_in_chord_counter = 0

        for event in flat:
                abs_offset = float(event.offset)
                curr_delta_offset = abs_offset - prev_offset
                prev_offset = abs_offset

                if isinstance(event, note.Note):
                    embedded_token_events.append((abs_offset, note_event(event, curr_instrument, curr_delta_offset)))
                    note_counter += 1
                elif isinstance(event, chord.Chord):
                    for i, n in enumerate(event.notes, start=0):
                        if i == 0:
                            chord_delta_offset = curr_delta_offset
                            embedded_token_events.append((abs_offset, note_event(n, curr_instrument, curr_delta_offset, part_of_chord=True, first_chord_note=True)))
                            note_in_chord_counter += 1
                            continue
                        embedded_token_events.append((abs_offset, (note_event(n, curr_instrument, chord_delta_offset, part_of_chord=True))))
                        note_in_chord_counter += 1
                    chord_counter += 1
                elif isinstance(event, note.Rest):
                    embedded_token_events.append((abs_offset, rest_event(event, curr_delta_offset)))
                    rest_counter += 1


        print(f"Events in embedded tokens: {len(embedded_token_events)}")
        print(f"Notes in embedded tokens: {note_counter}")
        print(f"Rests in embedded tokens: {rest_counter}")
        print(f"Chords in embedded tokens: {chord_counter}")
        print(f"Notes in chords in embedded tokens: {note_in_chord_counter}")

        ###########################################################################################################################################


        print("Start sorting...")

        embedded_token_events.sort(key=lambda x: x[0])

        # Extract just the embedded token event in time order
        embedded_token_events = [embedded_token_event for _, embedded_token_event in embedded_token_events]

        print("Finished sorting.")

        embedded_token_events.append(embedded_token_events)


        print("Finished encoding to tokens.", end="\r")


        self.extend_maps(embedded_token_events)
        return embedded_token_events

    # uses reversed map on integer list
    def detokenize(self, numerical_sequence : list[str]) -> stream.Stream: # token list for testing, but should become int list
        #
        #
        #

        print("Start detokenizing...", end="\r")

        s = stream.Stream()


        print("Finished detokenizing.")

        return s


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
