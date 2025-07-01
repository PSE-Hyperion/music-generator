from fractions import Fraction
from typing import List, Dict

from music21 import stream, note, chord
from music21.tempo import TempoIndication, MetronomeMark


class Sixtuple():
    #   Sixtuple note event featuring bar, position, pitch, duration, velocity, tempo
    #   Doesn't include instruments or time signature
    #   Bar could be limited to only 0-100 range (if dataset contains unreasonably long songs)
    #   Duration could be quantized, but only if necessary for dataset

    def __init__(self, bar : str, position : str, pitch : str, duration : str, velocity : str, tempo : str):
        self._bar = bar
        self._position = position
        self._pitch = pitch
        self._duration = duration
        self._velocity = velocity
        self._tempo = tempo

    @property
    def bar(self):
        return self._bar

    @property
    def position(self):
        return self._position

    @property
    def pitch(self):
        return self._pitch

    @property
    def duration(self):
        return self._duration

    @property
    def velocity(self):
        return self._velocity

    @property
    def tempo(self):
        return self._tempo


def detokenize(sixtuples : list[Sixtuple]) -> stream.Stream:
    #   Reconstructs a Stream from a list of sixtuples
    #   Rests are reconstructed implicitly from position gaps between note events

    print("Start detokenizing...")

    s = stream.Stream()
    current_offset = 0.0 # absolute
    current_tempo = None

    # Group events by position for chord reconstruction
    pending_notes: Dict[float, List[Sixtuple]] = {}

    for event in sixtuples:
        bar_num = int(event.bar.split('_')[1])
        position_16th = int(event.position.split('_')[1])

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
            tempo_value = int(events_at_position[0].tempo.split('_')[1])
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
            pitch_midi = int(event.pitch.split('_')[1])
            duration = float(Fraction(event.duration.split('_')[1]))
            velocity = int(event.velocity.split('_')[1])

            n = note.Note(midi=pitch_midi, quarterLength=duration)
            n.volume.velocity = velocity
            s.insert(abs_offset, n)

        else:
            # Chord:
            pitches = []
            duration = None
            velocity = None

            for event in events_at_position:
                pitch_midi = int(event.pitch.split('_')[1])
                pitches.append(pitch_midi)
                if duration is None:
                    duration = float(Fraction(event.duration.split('_')[1]))
                    velocity = int(event.velocity.split('_')[1])

            c = chord.Chord(pitches, quarterLength=duration)
            c.volume.velocity = velocity
            s.insert(abs_offset, c)

        # Update current offset to the end of this event
        event_duration = float(Fraction(events_at_position[0].duration.split('_')[1]))
        current_offset = abs_offset + event_duration

    print("Finished detokenizing.")
    return s


class SixtupleTokenMaps():
    def __init__(self):
        self._bar_map = {}
        self._position_map = {}
        self._pitch_map = {}
        self._duration_map = {}
        self._velocity_map = {}
        self._tempo_map = {}

    @property
    def bar_map(self) -> dict[str, int]:
        """
        Returns a copy of the dictionary.
        """

        return self._bar_map.copy()

    @property
    def position_map(self) -> dict[str, int]:
        """
        Returns a copy of the dictionary.
        """

        return self._position_map.copy()

    @property
    def pitch_map(self) -> dict[str, int]:
        """
        Returns a copy of the dictionary.
        """

        return self._pitch_map.copy()

    @property
    def duration_map(self) -> dict[str, int]:
        """
        Returns a copy of the dictionary.
        """

        return self._duration_map.copy()

    @property
    def velocity_map(self) -> dict[str, int]:
        """
        Returns a copy of the dictionary.
        """

        return self._velocity_map.copy()

    @property
    def tempo_map(self) -> dict[str, int]:
        """
        Returns a copy of the dictionary.
        """

        return self._tempo_map.copy()

    @property
    def total_size(self) -> int:
        return len(self._bar_map) + len(self._position_map) + len(self._pitch_map) + len(self._duration_map) + len(self._velocity_map) + len(self._tempo_map)

    @property
    def bar_map_size(self) -> int:
        return len(self._bar_map)

    @property
    def position_map_size(self) -> int:
        return len(self._position_map)

    @property
    def pitch_map_size(self) -> int:
        return len(self._pitch_map)

    @property
    def duration_map_size(self) -> int:
        return len(self._duration_map)

    @property
    def velocity_map_size(self) -> int:
        return len(self._velocity_map)

    @property
    def tempo_map_size(self) -> int:
        return len(self._tempo_map)

    def extend(self, sixtuples : list[Sixtuple]):
        """
        Since the tokenizer tokenizes in batches, this method is used to extend the maps of features of a sixtuple after every new tokenization. That way, the tokenizer
        keeps track of all unique sixtuple features across all tokenized scores. After having tokenized all scores, the maps can be saved with token_maps_io.py
        """

        print("Start extending maps of tokens...")
        for sixtuple in sixtuples:
            if sixtuple.bar not in self._bar_map:
                self._bar_map[sixtuple.bar] = len(self._bar_map)
            if sixtuple.position not in self._position_map:
                self._position_map[sixtuple.position] = len(self._position_map)
            if sixtuple.pitch not in self._pitch_map:
                self._pitch_map[sixtuple.pitch] = len(self._pitch_map)
            if sixtuple.duration not in self._duration_map:
                self._duration_map[sixtuple.duration] = len(self._duration_map)
            if sixtuple.velocity not in self._velocity_map:
                self._velocity_map[sixtuple.velocity] = len(self._velocity_map)
            if sixtuple.tempo not in self._tempo_map:
                self._tempo_map[sixtuple.tempo] = len(self._tempo_map)
        print("Finished extending maps of tokens.")


class Tokenizer():

    def __init__(self, processed_dataset_id : str):
        self.processed_dataset_id = processed_dataset_id

        self.sixtuple_token_maps = SixtupleTokenMaps()



    def tokenize(self, score : stream.Score) -> list[Sixtuple]:
        """
        Tokenizes music21 score object to a list of sixtuples

        The score is flattened and all valuable data is extracted and saved in sixtuples, which represent a note event

        Rests are encoded implicitly
        """

        print("Start encoding to tokens...")

        flat = score.flatten()
        sixtuples : List[Sixtuple] = []

        # Get tempo from score (default to 120 if not found)
        # Two classes could contain this data, so we have to check both
        tempo_indications = flat.getElementsByClass('TempoIndication')
        metronome_marks = flat.getElementsByClass('MetronomeMark')
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
                        tempo=f"TEMPO_{current_tempo}"
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
                            tempo=f"TEMPO_{current_tempo}"
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

        self.sixtuple_token_maps.extend(sixtuples)
        return sixtuples

