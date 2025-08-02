from fractions import Fraction
import logging

from mido import MidiFile
from music21 import chord, interval, key, note, pitch, stream
from music21.tempo import MetronomeMark, TempoIndication

from groove_panda.config import Config, TokenizeMode
from groove_panda.midi.sheet_music_generator import generate_sheet_music
from groove_panda.processing.tokenization import midi_file_utils

config = Config()
logger = logging.getLogger(__name__)


class Sixtuple:
    """
    Sixtuple note event featuring bar, position, pitch, duration, velocity, tempo

    Doesn't include instruments or time signature

    Bar could be limited to only 0-100 range (if dataset contains unreasonably long songs)

    Duration could be quantized, but only if necessary for dataset
    """

    BAR_PREFIX = "BAR_"
    POSITION_PREFIX = "POSITION_"
    PITCH_PREFIX = "PITCH_"
    DURATION_PREFIX = "DURATION_"
    VELOCITY_PREFIX = "VELOCITY_"
    TEMPO_PREFIX = "TEMPO_"

    def __init__(self, bar: str, position: str, pitch: str, duration: str, velocity: str, tempo: str):
        self._bar = Sixtuple.BAR_PREFIX + bar
        self._position = Sixtuple.POSITION_PREFIX + position
        self._pitch = Sixtuple.PITCH_PREFIX + pitch
        self._duration = Sixtuple.DURATION_PREFIX + duration
        self._velocity = Sixtuple.VELOCITY_PREFIX + velocity
        self._tempo = Sixtuple.TEMPO_PREFIX + tempo

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

    def __repr__(self):
        return self.bar + self.position + self.pitch + self.duration + self.velocity + self.tempo


def detokenize(sixtuples: list[Sixtuple]) -> stream.Stream:
    """
    Reconstructs a Stream from a list of sixtuples
    Rests are reconstructed implicitly from position gaps between note events
    """

    logger.info("Start detokenizing...")

    s = stream.Stream()
    current_offset = 0.0  # absolute
    current_tempo = None

    # Group events by position for chord reconstruction
    pending_notes: dict[float, list[Sixtuple]] = {}

    for event in sixtuples:
        try:
            bar_num = int(event.bar.split("_", 1)[1])
        except Exception:
            logger.error("Can't parse bar token: %s", repr(event.bar))
            raise
        try:
            position_16th = int(event.position.split("_", 1)[1])
        except Exception:
            logger.error("Can't parse position token: %s", repr(event.position))
            raise

        # Convert to absolute offset, assuming 4/4
        # This is so cool
        abs_offset = bar_num * 4.0 + position_16th / 4.0

        # Collect all notes, playing at the same absolute offset
        if abs_offset not in pending_notes:
            pending_notes[abs_offset] = []
        pending_notes[abs_offset].append(event)

    # Sort dictionary by its absolute offset keys, to iterate in correct order
    sorted_offsets = sorted(pending_notes.keys())

    # Big loop, inserting multiple events, if needed, per iteration into the stream
    for abs_offset in sorted_offsets:
        events_at_position = pending_notes[abs_offset]

        # Check if tempo has changed at this position
        if events_at_position:
            # I am not sure we need to round in detokenize, since tokens already only have rounded values - joao
            tempo_value = round_tempo(int(events_at_position[0].tempo.split("_")[1]))
            if current_tempo != tempo_value:
                current_tempo = tempo_value
                s.insert(abs_offset, TempoIndication(number=current_tempo))
                s.insert(abs_offset, MetronomeMark(number=current_tempo))

        # Add rest if there's a gap
        if abs_offset > current_offset:
            rest_duration = abs_offset - current_offset
            if rest_duration > 0:
                s.insert(current_offset, note.Rest(quarterLength=rest_duration))

        # Initialize event duration before loop.
        event_duration: float = 0

        """
        Add each event at the given position to the stream. There is no differentiation between notes and chords.
        Chords are implicitly included in the form of several individual notes starting at the same position.
        """
        for event in events_at_position:
            pitch_midi = int(event.pitch.split("_")[1])
            duration = float(Fraction(event.duration.split("_")[1]))
            velocity = int(event.velocity.split("_")[1])

            n = note.Note(midi=pitch_midi, quarterLength=duration)
            n.volume.velocity = velocity
            s.insert(abs_offset, n)

            """
            event_duration is used to calculate rests in between notes.
            In the case of several notes in one position, the rest should only start
            after the longest note has ended, since only then will there be silence.
            """
            event_duration = max(event_duration, duration)

        # Update current offset to the end of this event
        current_offset = abs_offset + event_duration

    logger.info("Finished detokenizing.")
    if config.create_sheet_music:
        generate_sheet_music(s)
    return s


class SixtupleTokenMaps:
    """
    Internal token map container for tokenizer, to avoid sharing the tokenizer with other files, but just a container,
    that savely returns the data by using copies.

    The tokenizer can use this container to extend the token maps during processing of a dataset
    """

    PITCH_MIN = 21
    PITCH_MAX = 109  # inclusive
    VELOCITY_MIN = 20
    VELOCITY_MAX = 130  # inclusive
    BAR_MIN = 0
    BAR_MAX = 200  # or dynamic, but must be fixed per model
    POSITION_QUANTIZATION = 16  # e.g. 16 positions per bar
    TEMPO_MIN = 30
    TEMPO_MAX = 300
    TEMPO_STEP = 1
    DURATION_MIN = 0.25
    DURATION_MAX = 10

    def __init__(self):
        self._bar_map = {}
        self._position_map = {}
        self._pitch_map = {}
        self._duration_map = {}
        self._velocity_map = {}
        self._tempo_map = {}

    def create_from_ranges(self):
        self._bar_map = {Sixtuple.BAR_PREFIX + str(i): i - self.BAR_MIN for i in range(self.BAR_MIN, self.BAR_MAX + 1)}
        self._position_map = {Sixtuple.POSITION_PREFIX + str(i): i for i in range(self.POSITION_QUANTIZATION)}
        self._pitch_map = {
            Sixtuple.PITCH_PREFIX + str(i): i - self.PITCH_MIN for i in range(self.PITCH_MIN, self.PITCH_MAX + 1)
        }
        self._velocity_map = {
            Sixtuple.VELOCITY_PREFIX + str(i): i - self.VELOCITY_MIN
            for i in range(self.VELOCITY_MIN, self.VELOCITY_MAX + 1)
        }
        self._tempo_map = {
            Sixtuple.TEMPO_PREFIX + str(i): (i - self.TEMPO_MIN) // self.TEMPO_STEP
            for i in range(self.TEMPO_MIN, self.TEMPO_MAX + 1, self.TEMPO_STEP)
        }
        # Duration is model specific: could be predefined bins
        self._duration_map = self._create_duration_map()

    def _create_duration_map(self) -> dict[str, int]:
        step = 0.25
        count = int((self.DURATION_MAX - self.DURATION_MIN) / step) + 1
        return {Sixtuple.DURATION_PREFIX + str(self.DURATION_MIN + i * step): i + 1 for i in range(count + 1)}

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


def round_tempo(tempo: int) -> int:
    return round(tempo / config.tempo_round_value) * config.tempo_round_value


def quantize(value: float, precision: float) -> float:
    return max(round(value / precision) * precision, precision)


class Tokenizer:
    def __init__(self, processed_dataset_id: str = "EMPTY"):
        self.processed_dataset_id = processed_dataset_id

        self.sixtuple_token_maps = SixtupleTokenMaps()

    def tokenize(self, parsed_midi: stream.Score | MidiFile) -> list[Sixtuple]:
        """
        Tokenizes the parsed midi according to it's mode
        """

        if config.tokenize_mode is TokenizeMode.ORIGINAL:
            sixtuples = self._tokenize_original_key(parsed_midi)
        elif config.tokenize_mode is TokenizeMode.ALL_KEYS:
            sixtuples = self._tokenize_all_keys(parsed_midi)
        elif config.tokenize_mode is TokenizeMode.C_MAJOR_A_MINOR:
            sixtuples = self._tokenize_cmajor_aminor(parsed_midi)
        else:
            raise ValueError(f"Unsupported TOKENIZE_MODE: {config.tokenize_mode!r}")

        return sixtuples

    def _tokenize_original_key(self, parsed_midi: stream.Score | MidiFile) -> list[Sixtuple]:
        """
        Tokenizes the parsed midi in its original key, according to the type of parsed midi
        """

        if isinstance(parsed_midi, stream.Score):
            return self._tokenize_original_key_score(parsed_midi)

        return self._tokenize_original_key_midi_file(parsed_midi)

    def _tokenize_all_keys(self, parsed_midi: stream.Score | MidiFile) -> list[Sixtuple]:
        """ """

        if isinstance(parsed_midi, stream.Score):
            return self._tokenize_all_keys_score(parsed_midi)

        return self._tokenize_all_keys_midi_file(parsed_midi)

    def _tokenize_cmajor_aminor(self, parsed_midi: stream.Score | MidiFile) -> list[Sixtuple]:
        """
        Transpose every piece to C major (if originally major) or A minor (if originally minor),
        and return the tokens from that single transposition as a list.
        """

        if isinstance(parsed_midi, stream.Score):
            return self._tokenize_cmajor_aminor_score(parsed_midi)

        return self._tokenize_cmajor_aminor_midi_file(parsed_midi)

    def _tokenize_original_key_score(self, score: stream.Score) -> list[Sixtuple]:
        """
        Tokenizes music21 stream.Score with orginal key
        """

        return self._tokenize_score(score)

    def _tokenize_all_keys_score(self, score: stream.Score) -> list[Sixtuple]:
        """
        Transpose a music21 stream.Score through all 12 semitone steps (0 = original, +1, ..., +11)
        and return a flat list of all tokens across every transposition.
        """

        all_tokens: list[Sixtuple] = []
        for semitone_shift in range(12):
            transposed_midi = score.transpose(semitone_shift)
            if transposed_midi:
                all_tokens.extend(self._tokenize_score(transposed_midi))
            else:
                raise Exception("Couldn't transpose score to semitone shift")
        return all_tokens

    def _tokenize_cmajor_aminor_score(self, parsed_midi: stream.Score) -> list[Sixtuple]:
        """
        Transpose of music21 stream.Score every piece to C major (if originally major) or A minor (if originally minor),
        and return the tokens from that single transposition as a list.
        """

        analyzed_key = parsed_midi.analyze("key")
        if isinstance(analyzed_key, key.Key):
            tonic: pitch.Pitch = analyzed_key.tonic

            target_tonic = pitch.Pitch("C") if analyzed_key.mode == "major" else pitch.Pitch("A")

            transposition_interval = interval.Interval(tonic, target_tonic)
            transposed_midi = parsed_midi.transpose(transposition_interval)

            if transposed_midi:
                return self._tokenize_score(transposed_midi)

            raise Exception("Transposition of score was unsuccessful and returned null.")
        raise Exception("Analyzing of score was unsuccessful and didn't return a key.")

    def _tokenize_original_key_midi_file(self, midi_file: MidiFile) -> list[Sixtuple]:
        """
        Tokenizes mido MidiFile with orginal key
        """

        return self._tokenize_midi_file(midi_file)

    def _tokenize_all_keys_midi_file(self, parsed_midi: MidiFile) -> list[Sixtuple]:
        """
        Transpose the score through all 12 semitone steps (0 = original, +1, ..., +11)
        and return a flat list of all tokens across every transposition.
        """

        all_tokens: list[Sixtuple] = []

        for semitone_shift in range(12):
            transposed_midi = midi_file_utils.transpose(parsed_midi, semitone_shift)
            if transposed_midi:
                all_tokens.extend(self._tokenize_midi_file(transposed_midi))
            else:
                raise Exception("Couldn't transpose MIDI to semitone shift")
        return all_tokens

    def _tokenize_cmajor_aminor_midi_file(self, parsed_midi: MidiFile) -> list[Sixtuple]:
        """
        Transpose every piece to C major (if originally major) or A minor (if originally minor),
        and return the tokens from that single transposition as a list.
        """

        if isinstance(parsed_midi, stream.Score):
            analyzed_key = parsed_midi.analyze("key")
            if isinstance(analyzed_key, key.Key):
                tonic: pitch.Pitch = analyzed_key.tonic

                target_tonic = pitch.Pitch("C") if analyzed_key.mode == "major" else pitch.Pitch("A")

                transposition_interval = interval.Interval(tonic, target_tonic)
                transposed_midi = parsed_midi.transpose(transposition_interval)

                if transposed_midi:
                    return self.tokenize(transposed_midi)

                raise Exception("Transposition of score was unsuccessful and returned null.")
            raise Exception("Analyzing of score was unsuccessful and didn't return a key.")

        # TODO Implement minor/major transposition for MidiFile. There seems to be no analyze aquivalent in mido
        logger.warning(
            "Tokenizer mode: C_MAJOR_A_MINOR, doesn't work with parser: MIDO. For more information "
            "read TODO in tokenizer.py _tokenize_cmajor_aminor_midi_file()"
        )
        return []

    def _tokenize_score(self, score: stream.Score) -> list[Sixtuple]:
        """
        Tokenizes music21 stream.Score object to a list of sixtuples.

        The score is flattened and all valuable data is extracted and saved in sixtuples, which represent a note event

        Rests are encoded implicitly
        """

        logger.info("Start encoding to tokens...")

        flat = score.flatten()
        sixtuples: list[Sixtuple] = []

        # Get tempo from score (default to 120 if not found)
        # Two classes could contain this data, so we have to check both
        tempo_indications = flat.getElementsByClass("TempoIndication")
        metronome_marks = flat.getElementsByClass("MetronomeMark")
        current_tempo = round_tempo(config.default_tempo)

        # Set first tempo
        if tempo_indications:
            current_tempo = round_tempo(int(tempo_indications[0].number))
            # logger.info("TempoIndication found: %s ", current_tempo)
        elif metronome_marks:
            current_tempo = round_tempo(int(metronome_marks[0].number))
            # logger.info("MetronomeMark found: %s",current_tempo)
        else:
            pass
            # logger.info("No tempo found, using default: %s", current_tempo)

        # Time signature is always 4/4 in our dataset
        beats_per_bar = 4

        tempo_changes = sorted(
            [(ti.offset, round_tempo(int(ti.number))) for ti in tempo_indications]
            + [(mm.offset, round_tempo(int(mm.number))) for mm in metronome_marks]
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

            while (
                tempo_idx < len(tempo_changes)
                and abs(tempo_changes[tempo_idx][0] - abs_offset) < config.tempo_tolerance
            ):
                current_tempo = tempo_changes[tempo_idx][1]
                tempo_idx += 1

            # Calculate bar and position
            bar_number = int(abs_offset // beats_per_bar)
            position_in_bar = abs_offset % beats_per_bar

            # Quantize position to 16th notes, since all songs from dataset are 4/4
            position_16th = int(position_in_bar * 4)

            if isinstance(event, note.Note):
                sixtuples.append(
                    Sixtuple(
                        bar=str(bar_number),
                        position=str(position_16th),
                        pitch=str(event.pitch.midi),
                        duration=str(quantize(float(event.quarterLength), 0.25)),
                        velocity=str(event.volume.velocity),
                        tempo=str(current_tempo),
                    )
                )
                note_counter += 1

            elif isinstance(event, chord.Chord):
                # Each note in the chord becomes a separate Sixtuple
                # They all share the same bar and position
                for chord_note in event.notes:
                    sixtuples.append(
                        Sixtuple(
                            bar=str(bar_number),
                            position=str(position_16th),
                            pitch=str(chord_note.pitch.midi),
                            duration=str(quantize(float(event.quarterLength), 0.25)),
                            velocity=str(event.volume.velocity),
                            tempo=str(current_tempo),
                        )
                    )
                    note_in_chord_counter += 1
                chord_counter += 1

            elif isinstance(event, note.Rest):
                # Rests are encoded implicitly from position gaps
                rest_counter += 1

        return sixtuples

    def _tokenize_midi_file(self, midi_file: MidiFile) -> list[Sixtuple]:
        """
        Tokenizes mido MidiFile object to a list of sixtuples.

        First calls read_and_merge_events to get all valuable information from the MidiFile.

        Then translates the returned result to a list of Sixtuple.

        Rests are encoded implicitly.
        """

        logger.info("Start encoding to tokens...")

        merged_events, ticks_per_beat = midi_file_utils.read_and_merge_events(midi_file)

        sixtuples: list[Sixtuple] = []
        active_notes: dict = {}

        current_tempo = 500000  # microseconds per beat, default = 120bpm
        qn_per_bar = 4  # quarter notes per bar

        for event in merged_events:
            tick = event["abs_tick"]

            # Tempo change
            if event["type"] == "set_tempo":
                current_tempo = event["tempo"]

            # Note on
            elif event["type"] == "note_on":
                active_notes[event["note"]].append((tick, event["velocity"], current_tempo))

            # Note off
            elif event["type"] == "note_off" and active_notes[event["note"]]:
                start_tick, velocity, tempo = active_notes[event["note"]].pop(0)
                duration_ticks = tick - start_tick
                duration_qn = duration_ticks / ticks_per_beat
                start_qn = start_tick / ticks_per_beat

                # Bar and position
                bar = int(start_qn // qn_per_bar)
                position_qn = start_qn % qn_per_bar
                position_16th = round(position_qn)

                sixtuples.append(
                    Sixtuple(
                        bar=str(bar),
                        position=str(position_16th),
                        pitch=str(event["note"]),
                        duration=str(round(duration_qn, 4)),
                        velocity=str(velocity),
                        tempo=str(round(60000000 / tempo)),
                    )
                )

        return sixtuples
