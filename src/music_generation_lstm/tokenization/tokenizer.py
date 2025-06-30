import json
import os

from fractions import Fraction
from typing import List

from music21 import stream, note, chord, instrument
from config import SEQUENCE_LENGTH, QUANTIZATION_PRECISION_DELTA_OFFSET, QUANTIZATION_PRECISION_DURATION, TOKEN_MAPS_DIR


class TokenEvent():
    def __init__(self, type : str, pitch : str, duration : str, delta_offset : str, velocity : str, instrument : str):
        self.type = type
        self.pitch = pitch
        self.duration = duration
        self.delta_offset = delta_offset
        self.velocity = velocity
        self.instrument = instrument

class PendingChordNote():
    def __init__(self, abs_offset : float, note : note.Note, chord_duration : float):
        self.abs_offset = abs_offset
        self.note = note
        self.chord_duration = chord_duration


def quantize(value, resolution : float = 1/8) -> float: #Any particular reason for 1/8?
    return round(value / resolution) * resolution

def note_event(note : note.Note, instrument : instrument.Instrument, curr_offset : float) -> TokenEvent:
    return TokenEvent(
        type="NOTE",
        pitch=f"{note.pitch}",
        duration=f"{quantize(value=float(Fraction(note.quarterLength)), resolution=QUANTIZATION_PRECISION_DURATION)}",
        delta_offset=f"{quantize(value=curr_offset, resolution=QUANTIZATION_PRECISION_DELTA_OFFSET)}",
        velocity=f"{note.volume.velocity}",
        instrument=f"{instrument.instrumentName}"
    )

def chord_note_event(chord_note : note.Note, chord_duration, instrument : instrument.Instrument, curr_offset : float, first_chord_note : bool = False) -> TokenEvent:
    # find chord_duration type (what is it)

    type = "CHORD_NOTE"
    if first_chord_note:
        type = "CHORD_NOTE_START"

    return TokenEvent(
        type=f"{type}",
        pitch=f"{chord_note.pitch}",
        # duration of chord note and of chord itself
        duration=f"{quantize(value=float(Fraction(chord_note.quarterLength)), resolution=QUANTIZATION_PRECISION_DURATION)}_{quantize(value=float(Fraction(chord_duration)), resolution=QUANTIZATION_PRECISION_DURATION)}",      # times time signature maybe ???
        delta_offset=f"{quantize(value=curr_offset, resolution=QUANTIZATION_PRECISION_DELTA_OFFSET)}",
        velocity=f"{chord_note.volume.velocity}",
        instrument=f"{instrument.instrumentName}"
    )

def rest_event(rest : note.Rest, curr_offset : float) -> TokenEvent:
    return TokenEvent(
        type="REST",
        pitch="NO_PITCH",
        duration=f"{quantize(value=float(Fraction(rest.quarterLength)), resolution=QUANTIZATION_PRECISION_DURATION)}",
        delta_offset=f"{quantize(value=curr_offset, resolution=QUANTIZATION_PRECISION_DELTA_OFFSET)}",
        velocity="NO_VELOCITY",
        instrument="NO_INSTRUMENT"
    )

def is_embedded_token_rest(embedded_token : TokenEvent) -> bool:
    return embedded_token.type == "REST"

def is_embedded_token_note(embedded_token : TokenEvent) -> bool:
    return embedded_token.type == "NOTE"

def is_embedded_token_part_of_chord(embedded_token : TokenEvent) -> bool:
    return embedded_token.type.startswith("CHORD_NOTE")

def is_first_chord_note(embedded_token : TokenEvent) -> bool:
    return embedded_token.type == "CHORD_NOTE_START"

def detokenize(embedded_token_events : list[TokenEvent]) -> stream.Stream:
    #
    #
    #

    print("Start detokenizing...")

    s = stream.Stream()
    pending_chord_notes : List[PendingChordNote] = []
    current_offset = 0.0  # absolute offset

    note_in_chord_counter = 0

    for embedded_token_event in embedded_token_events:
        if is_embedded_token_rest(embedded_token_event):
            dur = float(embedded_token_event.duration)
            delta_offset = float(embedded_token_event.delta_offset)
            current_offset += delta_offset
            s.insert(current_offset, note.Rest(quarterLength=dur))

        elif is_embedded_token_note(embedded_token_event):
            pitch = embedded_token_event.pitch
            dur = float(embedded_token_event.duration)
            delta_offset = float(embedded_token_event.delta_offset)
            velocity = int(embedded_token_event.velocity)
            current_offset += delta_offset
            n = note.Note(pitch, quarterLength=dur)
            n.volume.velocity = velocity
            s.insert(current_offset, n)

        elif is_embedded_token_part_of_chord(embedded_token_event):
            # Flush previous chord if this is a new chord start
            print(embedded_token_event.type)
            if len(pending_chord_notes) > 0 and is_first_chord_note(embedded_token_event):
                first_chord_note = pending_chord_notes[0]
                chord_offset = first_chord_note.abs_offset
                pitches = [pending_chord_note.note.pitch for pending_chord_note in pending_chord_notes]
                chord_duration = first_chord_note.chord_duration
                s.insert(chord_offset, chord.Chord(pitches, quarterLength=chord_duration))
                note_in_chord_counter += len(pitches)
                pending_chord_notes.clear()

            pitch = embedded_token_event.pitch
            dur, chord_dur = embedded_token_event.duration.split("_")
            dur = float(dur)
            chord_dur = float(chord_dur)
            delta_offset = float(embedded_token_event.delta_offset)
            velocity = int(embedded_token_event.velocity)

            if len(pending_chord_notes) == 0:
                current_offset += delta_offset  # Only update on first note of chord

            n = note.Note(pitch, quarterLength=dur)
            n.volume.velocity = velocity
            pending_chord_notes.append(PendingChordNote(current_offset, n, chord_dur))

    # Flush remaining chord notes
    if pending_chord_notes:
        first_chord_note = pending_chord_notes[0]
        chord_offset = first_chord_note.abs_offset
        pitches = [pending_chord_note.note.pitch for pending_chord_note in pending_chord_notes]
        chord_duration = first_chord_note.chord_duration
        s.insert(chord_offset, chord.Chord(pitches, quarterLength=chord_duration))
        note_in_chord_counter += len(pitches)
        pending_chord_notes.clear()


    print("Finished detokenizing.")

    return s

class Tokenizer():
    def __init__(self, processed_dataset_id : str):
        self.processed_dataset_id = processed_dataset_id

        self.type_map = {}
        self.pitch_map = {}
        self.duration_map = {}
        self.delta_offset_map = {}
        self.velocity_map = {}
        self.instrument_map = {}

        self.sequence_length = SEQUENCE_LENGTH
        self.num_features_type = 0
        self.num_features_pitch = 0
        self.num_features_duration = 0
        self.num_features_delta_offset = 0
        self.num_features_velocity = 0
        self.num_features_instrument = 0

    def extend_maps(self, embedded_token_events : list[TokenEvent]):
        #   Extends the maps of this tokenizer instance
        #   While using this tokenizer, the maps get updated for each new feature found in the given tokenized list of embedded token events
        #

        print("Start extending maps of tokens...")

        for event in embedded_token_events:
            if event.type not in self.type_map:
                self.type_map[event.type] = len(self.type_map)
            if event.pitch not in self.pitch_map:
                self.pitch_map[event.pitch] = len(self.pitch_map)
            if event.duration not in self.duration_map:
                self.duration_map[event.duration] = len(self.duration_map)
            if event.delta_offset not in self.delta_offset_map:
                self.delta_offset_map[event.delta_offset] = len(self.delta_offset_map)
            if event.velocity not in self.velocity_map:
                self.velocity_map[event.velocity] = len(self.velocity_map)
            if event.instrument not in self.instrument_map:
                self.instrument_map[event.instrument] = len(self.instrument_map)

        print("Finished extending maps of tokens.")

    def save_maps(self):
        #
        #
        #

        print("Start saving maps...")

        total_unique_tokens = len(self.type_map) + len(self.pitch_map)+len(self.duration_map)+len(self.delta_offset_map)+len(self.velocity_map)+len(self.instrument_map)

        print(f"Total unique tokens: {total_unique_tokens}")

        # save important information in tokenizer just in case, could also be saved in data
        self.sequence_length = SEQUENCE_LENGTH
        self.num_features_type = len(self.type_map)
        self.num_features_pitch = len(self.pitch_map)
        self.num_features_duration = len(self.duration_map)
        self.num_features_delta_offset = len(self.delta_offset_map)
        self.num_features_velocity = len(self.velocity_map)
        self.num_features_instrument = len(self.instrument_map)


        folder_path = os.path.join(TOKEN_MAPS_DIR, self.processed_dataset_id)
        os.makedirs(folder_path, exist_ok=False)
        with open(os.path.join(folder_path, "type_map.json"), "w") as f:
            json.dump(self.type_map, f, indent=4)
        with open(os.path.join(folder_path, "pitch_map.json"), "w") as f:
            json.dump(self.pitch_map, f, indent=4)
        with open(os.path.join(folder_path, "duration_map.json"), "w") as f:
            json.dump(self.duration_map, f, indent=4)
        with open(os.path.join(folder_path, "delta_offset_map.json"), "w") as f:
            json.dump(self.delta_offset_map, f, indent=4)
        with open(os.path.join(folder_path, "velocity_map.json"), "w") as f:
            json.dump(self.velocity_map, f, indent=4)
        with open(os.path.join(folder_path, "instrument_map.json"), "w") as f:
            json.dump(self.instrument_map, f, indent=4)

        print("Finished saving maps")

    def tokenize(self, score : stream.Score) -> list[TokenEvent]:
        #   Receives a score, that it will tokenize to embedded token events
        #   EmbeddedTokenEvents is a group of tokens per event
        #   The score is turned into a list of embedded token events

        print("Start encoding to tokens...")

        flat = score.flatten().notesAndRests

        embedded_token_events : List[TokenEvent] = []
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
                    embedded_token_events.append(note_event(event, curr_instrument, curr_delta_offset))
                    note_counter += 1
                elif isinstance(event, chord.Chord):
                    chord_duration = event.quarterLength
                    for i, n in enumerate(event.notes, start=0):
                        if i == 0:
                            chord_delta_offset = curr_delta_offset
                            embedded_token_events.append(chord_note_event(n, chord_duration, curr_instrument, curr_delta_offset, first_chord_note=True))
                        else:
                            embedded_token_events.append(chord_note_event(n, chord_duration, curr_instrument, chord_delta_offset))
                        print(embedded_token_events[-1].type) # TODO DELETE
                        note_in_chord_counter += 1
                    chord_counter += 1
                elif isinstance(event, note.Rest):
                    embedded_token_events.append(rest_event(event, curr_delta_offset))
                    rest_counter += 1

        # Optional print
        if False:
            print(f"Events in embedded tokens: {len(embedded_token_events)}")
            print(f"Notes in embedded tokens: {note_counter}")
            print(f"Rests in embedded tokens: {rest_counter}")
            print(f"Chords in embedded tokens: {chord_counter}")
            print(f"Note in chords in embedded tokens: {note_in_chord_counter}")


        self.extend_maps(embedded_token_events)
        return embedded_token_events

    @classmethod
    def load(cls):
        #
        #   load all variables declared in __init__()
        #

        pass

