from collections.abc import Iterable
from fractions import Fraction
import logging

from music21 import chord, note, stream
from music21.tempo import MetronomeMark, TempoIndication
import numpy as np
from numpy import uint8

from music_generation_lstm.processing.tokenization import tokens
from music_generation_lstm.processing.tokenization.tokens import HexTuple
from music_generation_lstm.step import pipeline_step

logger = logging.getLogger(__name__)
TEMPO_CHANGE_ERROR = 0.01

def detokenize(hex_tuples: list[HexTuple]) -> stream.Stream:
    """
    Reconstructs a Stream from a list of hextuples
    Rests are reconstructed implicitly from position gaps between note events
    """

    print("Start detokenizing...")

    s = stream.Stream()
    current_offset = 0.0  # absolute
    current_tempo = None

    # Group events by position for chord reconstruction
    pending_notes: dict[float, list[HexTuple]] = {}

    for event in hex_tuples:
        bar_num = int(event.bar)
        position_16th = int(event.position)

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
            tempo_value = int(events_at_position[0].tempo)
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
            pitch_midi = int(event.pitch)
            duration = int(Fraction(event.duration)) #FIXME length is not necessarily in 1/4
            velocity = int(event.velocity)

            n = note.Note(midi=pitch_midi, quarterLength=duration) #FIXME
            n.volume.velocity = velocity
            s.insert(abs_offset, n)
        else:
            # Chord:
            pitches = []
            duration = None
            velocity = None

            for event in events_at_position:
                pitch_midi = int(event.pitch)
                pitches.append(pitch_midi)
                if duration is None:
                    duration = int(event.duration) #FIXME
                    velocity = int(event.velocity)

            c = chord.Chord(pitches, quarterLength=duration) #FIXME
            c.volume.velocity = velocity
            s.insert(abs_offset, c)

        # Update current offset to the end of this event
        event_duration = int(events_at_position[0].duration) #FIXME
        current_offset = abs_offset + event_duration

    print("Finished detokenizing.")
    return s


class HexTupleTokenMap:
    """
    Internal token map container for tokenizer, to avoid sharing the tokenizer with other files, but just a container,
    that savely returns the data by using copies.

    The tokenizer can use this container to extend the token maps during processing of a dataset
    """

    def __init__(self):
        self._bar_map = {}
        self._position_map = {}
        self._pitch_map = {}
        self._duration_map = {}
        self._velocity_map = {}
        self._tempo_map = {}

    @property
    def bar_map(self) -> dict[uint8, int]:
        """
        Returns a copy of the dictionary.
        """

        return self._bar_map.copy()

    @property
    def position_map(self) -> dict[uint8, int]:
        """
        Returns a copy of the dictionary.
        """

        return self._position_map.copy()

    @property
    def pitch_map(self) -> dict[uint8, int]:
        """
        Returns a copy of the dictionary.
        """

        return self._pitch_map.copy()

    @property
    def duration_map(self) -> dict[uint8, int]:
        """
        Returns a copy of the dictionary.
        """

        return self._duration_map.copy()

    @property
    def velocity_map(self) -> dict[uint8, int]:
        """
        Returns a copy of the dictionary.
        """

        return self._velocity_map.copy()

    @property
    def tempo_map(self) -> dict[uint8, int]:
        """
        Returns a copy of the dictionary.
        """

        return self._tempo_map.copy()

    @property
    def total_size(self) -> int:
        return (
            len(self._bar_map)
            + len(self._position_map)
            + len(self._pitch_map)
            + len(self._duration_map)
            + len(self._velocity_map)
            + len(self._tempo_map)
        )

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

    def extend(self, hex_tuples: Iterable[HexTuple]):
        """
        Since the tokenizer tokenizes in batches,
        this method is used to extend the maps of features of a hextuple after every new tokenization.
        That way, the tokenizer keeps track of all unique hextuples features across all tokenized scores.
        After having tokenized all scores, the maps can be saved with token_maps_io.py
        """

        print("Start extending maps of tokens...")
        for hex_tuple in hex_tuples:
            if hex_tuple.bar not in self._bar_map:
                self._bar_map[hex_tuple.bar] = len(self._bar_map)
            if hex_tuple.position not in self._position_map:
                self._position_map[hex_tuple.position] = len(self._position_map)
            if hex_tuple.pitch not in self._pitch_map:
                self._pitch_map[hex_tuple.pitch] = len(self._pitch_map)
            if hex_tuple.duration not in self._duration_map:
                self._duration_map[hex_tuple.duration] = len(self._duration_map)
            if hex_tuple.velocity not in self._velocity_map:
                self._velocity_map[hex_tuple.velocity] = len(self._velocity_map)
            if hex_tuple.tempo not in self._tempo_map:
                self._tempo_map[hex_tuple.tempo] = len(self._tempo_map)
        print("Finished extending maps of tokens.")

    def create_from_sets(
        self, bar_set: set, position_set: set, pitch_set: set, duration_set: set, velocity_set: set, tempo_set: set
    ):
        self._bar_map = {token: idx for idx, token in enumerate(bar_set)}
        self._position_map = {token: idx for idx, token in enumerate(position_set)}
        self._pitch_map = {token: idx for idx, token in enumerate(pitch_set)}
        self._duration_map = {token: idx for idx, token in enumerate(duration_set)}
        self._velocity_map = {token: idx for idx, token in enumerate(velocity_set)}
        self._tempo_map = {token: idx for idx, token in enumerate(tempo_set)}

@pipeline_step(kind='tokenizer')
def tokenize_batch(scores: Iterable[stream.Score]) -> list[tokens.HexTuple]:
    token_sequences = []
    idx = -1

    for idx, s in enumerate(scores):
        logger.debug("Tokenizing item %s", idx + 1)
        token_sequence = tokenize(s)
        if token_sequence is not None:
            token_sequences.append(token_sequence)

    return token_sequences

def tokenize(score: stream.Score) -> list[tokens.HexTuple]:
    """
    Tokenizes music21 score object to a list of hextuples

    The score is flattened and all valuable data is extracted and saved in hextuples, which represent a note event

    Rests are encoded implicitly
    """

    flat = score.flatten()
    hextuples: list[tokens.HexTuple] = []

    # Get tempo from score (default to 120 if not found)
    # Two classes could contain this data, so we have to check both
    tempo_indications = flat.getElementsByClass("TempoIndication")
    metronome_marks = flat.getElementsByClass("MetronomeMark")
    current_tempo = 120

    # Set first tempo
    if tempo_indications:
        current_tempo = int(tempo_indications[0].number)
        # print(f"TempoIndication found: {current_tempo}")
    elif metronome_marks:
        current_tempo = int(metronome_marks[0].number)
        # print(f"MetronomeMark found: {current_tempo}")
    else:
        pass
        # print(f"No tempo found, using default: {current_tempo}")

    # Time signature is always 4/4 in our dataset
    beats_per_bar = 4

    tempo_changes = sorted(
        [(ti.offset, int(ti.number)) for ti in tempo_indications]
        + [(mm.offset, int(mm.number)) for mm in metronome_marks]
    )

    # Use an index to track which tempo is active
    tempo_idx = 0

    note_counter = 0
    rest_counter = 0
    chord_counter = 0
    note_in_chord_counter = 0

    # Big loop, that goes through all events and finds tempo changes, bar,
    # position and the notes itself (and the note's information)
    for event in flat:
        abs_offset = float(event.offset)

        while tempo_idx < len(tempo_changes) and abs(tempo_changes[tempo_idx][0] - abs_offset) < TEMPO_CHANGE_ERROR:
            current_tempo = tempo_changes[tempo_idx][1]
            tempo_idx += 1

        # Calculate bar and position
        bar_number = int(abs_offset // beats_per_bar)
        position_in_bar = abs_offset % beats_per_bar

        # Quantize position to 16th notes, since all songs from dataset are 4/4
        position_16th = int(position_in_bar * 4)

        if isinstance(event, note.Note):
            hextuples.append(
                tokens.HexTuple(
                    pitch=np.uint8(event.pitch.midi),
                    bar=np.uint8(bar_number),
                    position=np.uint8(position_16th),
                    duration=np.uint8(event.quarterLength * 8),
                    velocity=np.uint8(event.volume.velocity),
                    tempo=np.uint8(current_tempo),
                )
            )
            note_counter += 1

        elif isinstance(event, chord.Chord):
            # Each note in the chord becomes a separate hextuples
            # They all share the same bar and position
            for chord_note in event.notes:
                hextuples.append(
                    tokens.HexTuple(
                        pitch=np.uint8(chord_note.pitch.midi),
                        bar=np.uint8(bar_number),
                        position=np.uint8(position_16th),
                        duration=np.uint8(event.quarterLength * 8),
                        velocity=np.uint8(event.volume.velocity),
                        tempo=np.uint8(current_tempo),
                    )
                )
                note_in_chord_counter += 1
            chord_counter += 1

        elif isinstance(event, note.Rest):
            # Rests are encoded implicitly from position gaps
            rest_counter += 1

    # Delete for parallel processing
    return hextuples
