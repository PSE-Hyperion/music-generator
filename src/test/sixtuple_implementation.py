from fractions import Fraction
import os
from typing import Dict, List

from music21 import chord, note, stream
from music21.stream import Score
from music21.tempo import MetronomeMark, TempoIndication


class Sixtuple:
    #   Sixtuple note event featuring bar, position, pitch, duration, velocity, tempo
    #   Doesn't include instruments or time signature
    #   Bar could be limited to only 0-100 range (if dataset contains unreasonably long songs)
    #   Duration could be quantized, but only if necessary for dataset

    def __init__(self, bar: str, position: str, pitch: str, duration: str, velocity: str, tempo: str):
        self.bar = bar
        self.position = position
        self.pitch = pitch
        self.duration = duration
        self.velocity = velocity
        self.tempo = tempo


class PendingChordNote:
    def __init__(self, abs_offset: float, note: note.Note, chord_duration: float):
        self.abs_offset = abs_offset
        self.note = note
        self.chord_duration = chord_duration


def detokenize(sixtuples: list[Sixtuple]) -> stream.Stream:
    #   Reconstructs a Stream from a list of sixtuples
    #   Rests are reconstructed implicitly from position gaps between note events

    print("Start detokenizing...")

    s = stream.Stream()
    current_offset = 0.0  # absolute
    current_tempo = None

    # Group events by position for chord reconstruction
    pending_notes: Dict[float, List[Sixtuple]] = {}

    for event in sixtuples:
        bar_num = int(event.bar.split("_")[1])
        position_16th = int(event.position.split("_")[1])

        # Convert to absolute offset, assuming 4/4
        # Das ist so schön
        abs_offset = bar_num * 4.0 + position_16th / 4.0

        # Collect all notes, playing at the same absolute offset
        if abs_offset not in pending_notes:
            pending_notes[abs_offset] = []
        pending_notes[abs_offset].append(event)

    # Sort dictionary by it's absolute offset keys, to iterate in correct order
    sorted_offsets = sorted(pending_notes.keys())

    # Big loop, inserting multiple events, if needed, per iteration into the stream
    for abs_offset in sorted_offsets:
        events_at_position = pending_notes[abs_offset]

        # Check if tempo has changed at this position
        if events_at_position:
            tempo_value = int(events_at_position[0].tempo.split("_")[1])
            if current_tempo != tempo_value:
                current_tempo = tempo_value
                s.insert(abs_offset, TempoIndication(number=current_tempo))
                s.insert(abs_offset, MetronomeMark(number=current_tempo))

        # Add rest if there's a gap
        if abs_offset > current_offset:
            rest_duration = abs_offset - current_offset
            if rest_duration > 0:
                s.insert(current_offset, note.Rest(quarterLength=rest_duration))

        # Big if single note or chord, consisting of multiple notes
        # Single note:
        if len(events_at_position) == 1:
            event = events_at_position[0]
            pitch_midi = int(event.pitch.split("_")[1])
            duration = float(Fraction(event.duration.split("_")[1]))
            velocity = int(event.velocity.split("_")[1])

            n = note.Note(midi=pitch_midi, quarterLength=duration)
            n.volume.velocity = velocity
            s.insert(abs_offset, n)

        else:
            # Chord:
            pitches = []
            duration = None
            velocity = None

            for event in events_at_position:
                pitch_midi = int(event.pitch.split("_")[1])
                pitches.append(pitch_midi)
                if duration is None:
                    duration = float(Fraction(event.duration.split("_")[1]))
                    velocity = int(event.velocity.split("_")[1])

            c = chord.Chord(pitches, quarterLength=duration)
            c.volume.velocity = velocity
            s.insert(abs_offset, c)

        # Update current offset to the end of this event
        event_duration = float(Fraction(events_at_position[0].duration.split("_")[1]))
        current_offset = abs_offset + event_duration

    print("Finished detokenizing.")
    return s


class Tokenizer:
    def __init__(self, processed_dataset_id: str):
        self.processed_dataset_id = processed_dataset_id

        self.bar_map = {}
        self.position_map = {}
        self.pitch_map = {}
        self.duration_map = {}
        self.velocity_map = {}
        self.tempo_map = {}

    def extend_maps(self, sixtuples: list[Sixtuple]):
        #   extends the maps of this tokenizer instance
        #
        #

        print("Start extending maps of tokens...")

        for event in sixtuples:
            if event.bar not in self.bar_map:
                self.bar_map[event.bar] = len(self.bar_map)
            if event.position not in self.position_map:
                self.position_map[event.position] = len(self.position_map)
            if event.pitch not in self.pitch_map:
                self.pitch_map[event.pitch] = len(self.pitch_map)
            if event.duration not in self.duration_map:
                self.duration_map[event.duration] = len(self.duration_map)
            if event.velocity not in self.velocity_map:
                self.velocity_map[event.velocity] = len(self.velocity_map)
            if event.tempo not in self.tempo_map:
                self.tempo_map[event.tempo] = len(self.tempo_map)

        print("Finished extending maps of tokens.")

        print()
        print(f"TOTAL FEATURE COUNTS")
        print(f"Total sixtuples: {len(sixtuples)}")
        print(f"Total tokens: {len(sixtuples) * 6}")
        print()
        print(f"UNIQUE TOKEN FEATURE COUNTS")
        print(f"Unique bars: {len(self.bar_map)}")
        print(f"Unique positions: {len(self.position_map)}")
        print(f"Unique pitches: {len(self.pitch_map)}")
        print(f"Unique durations: {len(self.duration_map)}")
        print(f"Unique velocities: {len(self.velocity_map)}")
        print(f"Unique tempos: {len(self.tempo_map)}")
        print(
            f"Total unique tokens: {len(self.bar_map) + len(self.position_map) + len(self.pitch_map) + len(self.duration_map) + len(self.velocity_map) + len(self.tempo_map)}"
        )
        print()

    def tokenize(self, score: stream.Score) -> list[Sixtuple]:
        #   Tokenizes music21 score object to a list of sixtuple
        #   The score is flattened and all valuable data is extracted and saved in a sixtuple, which represents a note event
        #   Rests are encoded implicitly

        print("Start encoding to tokens...")

        flat = score.flatten()
        sixtuples: List[Sixtuple] = []

        # Get tempo from score (default to 120 if not found)
        # Two classes could contain this data, so we have to check both
        tempo_indications = flat.getElementsByClass("TempoIndication")
        metronome_marks = flat.getElementsByClass("MetronomeMark")
        current_tempo = 120

        # Set first tempo
        if tempo_indications:
            current_tempo = int(tempo_indications[0].number)
            print(f"TempoIndication found: {current_tempo}")
        elif metronome_marks:
            current_tempo = int(metronome_marks[0].number)
            print(f"MetronomeMark found: {current_tempo}")
        else:
            print(f"No tempo found, using default: {current_tempo}")

        # Time signature is always 4/4 in our dataset
        beats_per_bar = 4

        note_counter = 0
        rest_counter = 0
        chord_counter = 0
        note_in_chord_counter = 0

        # Big loop, that goes through all events and finds tempo changes, bar,
        # position and the notes itself (and the note's information)
        for event in flat:
            abs_offset = float(event.offset)

            # Again two checks for tempo in tempo indication and metronome marks
            # The loops find tempo indications, that are at the current event's offset (or very close),
            # since we encode tempo in sixtuples (note events)
            for tempo_indication in tempo_indications:
                if abs(tempo_indication.offset - abs_offset) < 0.01:  # Small tolerance
                    current_tempo = int(tempo_indication.number)

            for metronome_mark in metronome_marks:
                if abs(metronome_mark.offset - abs_offset) < 0.01:  # Small tolerance
                    current_tempo = int(metronome_mark.number)

            # Calculate bar and position
            bar_number = int(abs_offset // beats_per_bar)
            position_in_bar = abs_offset % beats_per_bar

            # Quantize position to 16th notes, since all songs from dataset are 4/4
            position_16th = int(position_in_bar * 4)

            if isinstance(event, note.Note):
                sixtuples.append(
                    Sixtuple(
                        bar=f"BAR_{bar_number}",
                        position=f"POSITION_{position_16th}",
                        pitch=f"PITCH_{event.pitch.midi}",
                        duration=f"DURATION_{event.quarterLength}",
                        velocity=f"VELOCITY_{event.volume.velocity}",
                        tempo=f"TEMPO_{current_tempo}",
                    )
                )
                note_counter += 1

            elif isinstance(event, chord.Chord):
                # Each note in the chord becomes a separate Sixtuple
                # They all share the same bar and position
                for chord_note in event.notes:
                    sixtuples.append(
                        Sixtuple(
                            bar=f"BAR_{bar_number}",
                            position=f"POSITION_{position_16th}",
                            pitch=f"PITCH_{chord_note.pitch.midi}",
                            duration=f"DURATION_{event.quarterLength}",
                            velocity=f"VELOCITY_{event.volume.velocity}",
                            tempo=f"TEMPO_{current_tempo}",
                        )
                    )
                    note_in_chord_counter += 1
                chord_counter += 1

            elif isinstance(event, note.Rest):
                # Rests are encoded implicitly from position gaps
                rest_counter += 1

        if False:
            print(f"Events in sixtuples: {len(sixtuples)}")
            print(f"Notes in sixtuples: {note_counter}")
            print(f"Chords in sixtuples: {chord_counter}")
            print(f"Note in chords in sixtuples: {note_in_chord_counter}")
            print(f"Rests skipped (implicit): {rest_counter}")

        self.extend_maps(sixtuples)
        return sixtuples


if __name__ == "__main__":
    from music21 import converter

    name = "0"
    midi_path = os.path.join("data", "midi", "datasets", "test", name) + ".mid"
    score = converter.parse(midi_path)

    if isinstance(score, Score):
        tokenizer = Tokenizer("")
        tokens = tokenizer.tokenize(score)
        reconstructed = detokenize(tokens)

        reconstructed_file_name = f"reconstructed_{name}_sixtuple.mid"
        u_input = input(f"Save {reconstructed_file_name} (y/n)?")
        if u_input == "y":
            reconstructed.write("midi", os.path.join("data", "midi", "results", reconstructed_file_name))
            print(f"Saved reconstructed_{name}_sixtuple.mid")
