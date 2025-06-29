import json
import os

from fractions import Fraction
from typing import List

from music21 import stream, note, chord, instrument
from config import SEQUENCE_LENGTH, QUANTIZATION_PRECISION_DELTA_OFFSET, QUANTIZATION_PRECISION_DURATION


class EmbeddedTokenEvent():
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


def quantize(value, resolution : float = 1/8) -> float:
    return round(value / resolution) * resolution

def note_event(note : note.Note, instrument : instrument.Instrument, curr_offset : float) -> EmbeddedTokenEvent:
    return EmbeddedTokenEvent(
        type="NOTE",
        pitch=f"{note.pitch}",
        duration=f"{quantize(value=float(Fraction(note.quarterLength)), resolution=QUANTIZATION_PRECISION_DURATION)}",
        delta_offset=f"{quantize(value=curr_offset, resolution=QUANTIZATION_PRECISION_DELTA_OFFSET)}",
        velocity=f"{note.volume.velocity}",
        instrument=f"{instrument.instrumentName}"
    )

def chord_note_event(chord_note : note.Note, chord_duration, instrument : instrument.Instrument, curr_offset : float, first_chord_note : bool = False) -> EmbeddedTokenEvent:
    # find chord_duration type (what is it)

    type = "CHORD_NOTE"
    if first_chord_note:
        type = "CHORD_NOTE_START"

    return EmbeddedTokenEvent(
        type=f"{type}",
        pitch=f"{chord_note.pitch}",
        # duration of chord note and of chord itself
        duration=f"{quantize(value=float(Fraction(chord_note.quarterLength)), resolution=QUANTIZATION_PRECISION_DURATION)}_{quantize(value=float(Fraction(chord_duration)), resolution=QUANTIZATION_PRECISION_DURATION)}",      # times time signature maybe ???
        delta_offset=f"{quantize(value=curr_offset, resolution=QUANTIZATION_PRECISION_DELTA_OFFSET)}",
        velocity=f"{chord_note.volume.velocity}",
        instrument=f"{instrument.instrumentName}"
    )

def rest_event(rest : note.Rest, curr_offset : float) -> EmbeddedTokenEvent:
    return EmbeddedTokenEvent(
        type="REST",
        pitch="NO_PITCH",
        duration=f"{quantize(value=float(Fraction(rest.quarterLength)), resolution=QUANTIZATION_PRECISION_DURATION)}",
        delta_offset=f"{quantize(value=curr_offset, resolution=QUANTIZATION_PRECISION_DELTA_OFFSET)}",
        velocity="NO_VELOCITY",
        instrument="NO_INSTRUMENT"
    )

def is_embedded_token_rest(embedded_token : EmbeddedTokenEvent) -> bool:
    return embedded_token.type == "REST"

def is_embedded_token_note(embedded_token : EmbeddedTokenEvent) -> bool:
    return embedded_token.type == "NOTE"

def is_embedded_token_part_of_chord(embedded_token : EmbeddedTokenEvent) -> bool:
    return embedded_token.type.startswith("CHORD_NOTE")

def is_first_chord_note(embedded_token : EmbeddedTokenEvent) -> bool:
    return embedded_token.type == "CHORD_NOTE_START"

def detokenize(embedded_token_events : list[EmbeddedTokenEvent]) -> stream.Stream:
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

class Maps():
    def __init__(self):
        self.type = {}
        self.pitch = {}
        self.duration = {}
        self.delta_offset = {}
        self.velocity = {}
        self.instrument = {}

    #aus tokenizer, aufrufe anpasssen, soll in maps_io
    def extend(self, embedded_token_events : list[EmbeddedTokenEvent]):
        #   extends the maps of this tokenizer instance
        #
        #
        print("Start extending maps of tokens...")
        for ev in embedded_token_events:
            if ev.type not in self.type:
                self.type[ev.type] = len(self.type)
            if ev.pitch not in self.pitch:
                self.pitch[ev.pitch] = len(self.pitch)
            if ev.duration not in self.duration:
                self.duration[ev.duration] = len(self.duration)
            if ev.delta_offset not in self.delta:
                self.delta[ev.delta_offset] = len(self.delta)
            if ev.velocity not in self.velocity:
                self.velocity[ev.velocity] = len(self.velocity)
            if ev.instrument not in self.instr:
                self.instr[ev.instrument] = len(self.instr)
        print("Finished extending maps of tokens.")


class Tokenizer():

    def __init__(self, processed_dataset_id : str):
        self.processed_dataset_id = processed_dataset_id

        self.sequence_length = SEQUENCE_LENGTH
        self.num_features_type = 0
        self.num_features_pitch = 0
        self.num_features_duration = 0
        self.num_features_delta_offset = 0
        self.num_features_velocity = 0
        self.num_features_instrument = 0

        self.maps.__init__


    def tokenize(self, score : stream.Score) -> list[EmbeddedTokenEvent]:   # return list of EmbeddedTokenEvents
        #   receives a score, that it will tokenize to EmbeddedTokenEvents
        #   EmbeddedTokenEvents is a group of tokens per event
        #   The score is turned into a list of embedded token events

        print("Start encoding to tokens...")

        flat = score.flatten().notesAndRests

        embedded_token_events : List[EmbeddedTokenEvent] = []
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


        self.maps.extend_maps(embedded_token_events)
        return embedded_token_events

    @classmethod
    def load(cls):
        #
        #   load all variables declared in __init__()
        #

        pass

