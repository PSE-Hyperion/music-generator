import json
import os

from music21 import converter
import numpy as np
from tensorflow.keras.models import Model  # type: ignore

from music_generation_lstm.config import GENERATION_LENGTH, SEQUENCE_LENGTH, TOKEN_MAPS_DIR
from music_generation_lstm.processing.tokenization.tokenizer import Sixtuple, Tokenizer, detokenize

# Constants for music timing
POSITIONS_PER_BAR = 16  # 16th note positions in 4/4 time
POSITION_MULTIPLIER = 4  # Convert quarter note duration to 16th note positions

# Constants for validation ranges
MAX_BAR_NUMBER = 400
MAX_POSITION_NUMBER = 15
MIN_MIDI_PITCH = 21
MAX_MIDI_PITCH = 108
MIN_NOTE_DURATION = 0.125
MAX_NOTE_DURATION = 4.0
MIN_MIDI_VELOCITY = 1
MAX_MIDI_VELOCITY = 127
MIN_TEMPO = 60
MAX_TEMPO = 200


class MusicGenerator:
    def __init__(self, model: Model, processed_dataset_id: str, temperature: float = 1.0):
        self.model = model
        self.processed_dataset_id = processed_dataset_id
        self.temperature = temperature

        # Load token maps for this dataset
        self._load_token_maps()

        # Create reverse mappings for converting indices back to tokens
        self._create_reverse_mappings()

    def _load_token_maps(self):
        """
        Load token maps and metadata from the processed dataset
        """

        print(f"Loading token maps for dataset: {self.processed_dataset_id}")

        token_maps_dir = os.path.join(TOKEN_MAPS_DIR, self.processed_dataset_id)

        # Load metadata
        metadata_path = os.path.join(token_maps_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Token maps metadata not found: {metadata_path}")

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # Load individual token maps
        self.token_maps = {}
        map_files = [
            ("bar", "bar_map.json"),
            ("position", "position_map.json"),
            ("pitch", "pitch_map.json"),
            ("duration", "duration_map.json"),
            ("velocity", "velocity_map.json"),
            ("tempo", "tempo_map.json"),
        ]

        for feature_name, filename in map_files:
            map_path = os.path.join(token_maps_dir, filename)
            if not os.path.exists(map_path):
                raise FileNotFoundError(f"Token map not found: {map_path}")

            with open(map_path) as f:
                self.token_maps[feature_name] = json.load(f)

        print("Token maps loaded successfully")

    def _create_reverse_mappings(self):
        """
        Create reverse mappings from indices to tokens for each feature
        """

        self.reverse_mappings = {}

        for feature_name, token_map in self.token_maps.items():
            # Create reverse mapping: index -> token
            self.reverse_mappings[feature_name] = {v: k for k, v in token_map.items()}

        print("Reverse mappings created")

    def _apply_temperature_sampling(self, probabilities: np.ndarray, temperature: float) -> int:
        """
        Apply temperature sampling to the predicted probabilities
        """

        if temperature == 0:
            # Greedy sampling (deterministic)
            return np.argmax(probabilities)

        # Apply temperature scaling
        scaled_logits = np.log(probabilities + 1e-8) / temperature

        # Apply softmax to get new probabilities
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        new_probabilities = exp_logits / np.sum(exp_logits)

        # Sample from the new distribution
        return np.random.choice(len(new_probabilities), p=new_probabilities)

    def _sequence_to_model_input(self, sequence: list[tuple[int, int, int, int, int, int]]) -> dict[str, np.ndarray]:
        """
        Convert a sequence of numeric sixtuples to model input format
        """

        # Convert to numpy array
        seq_array = np.array(sequence)

        # Create input dictionary for the model
        feature_names = ["bar", "position", "pitch", "duration", "velocity", "tempo"]

        model_input = {}
        for i, feature_name in enumerate(feature_names):
            # Add batch dimension (1, sequence_length)
            model_input[feature_name] = seq_array[:, i].reshape(1, -1)

        return model_input

    def _predictions_to_sixtuple(self, predictions: list[np.ndarray]) -> Sixtuple:
        """
        Convert model predictions to a Sixtuple using temperature sampling
        """

        feature_names = ["bar", "position", "pitch", "duration", "velocity", "tempo"]
        sampled_indices = []

        # Apply temperature sampling to each feature
        for pred in predictions:
            # pred is shape (1, vocab_size)
            probabilities = pred[0]
            sampled_idx = self._apply_temperature_sampling(probabilities, self.temperature)
            sampled_indices.append(sampled_idx)

        # Convert indices back to tokens
        tokens = []
        for i, feature_name in enumerate(feature_names):
            idx = sampled_indices[i]
            token = self.reverse_mappings[feature_name][idx]
            # Remove the prefix to get the raw value
            raw_value = token.split("_", 1)[1]
            tokens.append(raw_value)

        # Create Sixtuple
        return Sixtuple(
            bar=tokens[0], position=tokens[1], pitch=tokens[2], duration=tokens[3], velocity=tokens[4], tempo=tokens[5]
        )

    def _sixtuple_to_numeric(self, sixtuple: Sixtuple) -> tuple[int, int, int, int, int, int]:
        """
        Convert a Sixtuple back to numeric format
        """

        feature_names = ["bar", "position", "pitch", "duration", "velocity", "tempo"]
        tokens = [sixtuple.bar, sixtuple.position, sixtuple.pitch, sixtuple.duration, sixtuple.velocity, sixtuple.tempo]

        numeric_values = []
        for i, feature_name in enumerate(feature_names):
            token = tokens[i]
            idx = self.token_maps[feature_name][token]
            numeric_values.append(idx)

        return tuple(numeric_values)

    def generate_sequence(
        self, seed_sequence: list[tuple[int, int, int, int, int, int]], generation_length: int = GENERATION_LENGTH
    ) -> list[list[Sixtuple]]:
        """
        Generate a sequence of music events using the trained model.
        """
        print(f"Starting generation with temperature: {self.temperature}")
        print(f"Generation length: {generation_length}")

        if len(seed_sequence) < SEQUENCE_LENGTH:
            raise ValueError(f"Seed sequence must be at least {SEQUENCE_LENGTH} events long")

        current_sequence = seed_sequence[-SEQUENCE_LENGTH:]
        generated_events = []

        # Chord grouping: for each time step, generate 1 or more notes (randomly, but at least 1)
        for i in range(generation_length):
            model_input = self._sequence_to_model_input(current_sequence)
            predictions = self.model.predict(model_input, verbose=0)

            # Decide how many notes in this chord (1-3, random, but can be tuned)
            num_notes = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
            chord = []
            used_pitches = set()
            for _ in range(num_notes):
                sixtuple = self._predictions_to_sixtuple(predictions)
                # Avoid duplicate pitches in the same chord
                if sixtuple.pitch in used_pitches:
                    continue
                used_pitches.add(sixtuple.pitch)
                chord.append(sixtuple)
            if not chord:
                # fallback: always at least one note
                chord.append(self._predictions_to_sixtuple(predictions))
            generated_events.append(chord)

            # For next step, pick one of the chord notes as the next context (or average, but keep simple)
            numeric_event = self._sixtuple_to_numeric(chord[0])
            current_sequence = [*current_sequence[1:], numeric_event]

            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{generation_length} events")

        print("Generation completed")
        return generated_events

    def _normalize_generated_chords(self, generated_chords: list[list[Sixtuple]]) -> list[Sixtuple]:
        """
        Normalize generated chords to create a continuous sequence starting from bar 0.
        Flattens chords into a list of Sixtuples, assigning the same bar/position to all notes in a chord.
        """
        if not generated_chords:
            return []

        print("Normalizing generated sequence")
        normalized_sixtuples = []
        current_bar = 0
        current_position = 0

        for chord in generated_chords:
            # Each chord is a list of Sixtuples to be played at the same time
            for sixtuple in chord:
                normalized_sixtuple = Sixtuple(
                    bar=str(current_bar),
                    position=str(current_position),
                    pitch=sixtuple.pitch.split("_")[1],
                    duration=sixtuple.duration.split("_")[1],
                    velocity=sixtuple.velocity.split("_")[1],
                    tempo=sixtuple.tempo.split("_")[1],
                )
                normalized_sixtuples.append(normalized_sixtuple)

            # Advance position by the max duration in the chord
            max_duration = max(float(s.duration.split("_")[1]) for s in chord)
            position_advance = max(1, int(max_duration * POSITION_MULTIPLIER))
            current_position += position_advance
            if current_position >= POSITIONS_PER_BAR:
                current_bar += 1
                current_position = 0

        print(f"Normalization completed: {len(normalized_sixtuples)} events across {current_bar + 1} bars")
        return normalized_sixtuples

    def set_temperature(self, temperature: float):
        """
        Set the temperature for generation
        """

        self.temperature = temperature
        print(f"Temperature set to: {self.temperature}")

    def _validate_generated_sequence(self, generated_sixtuples: list[Sixtuple]) -> list[Sixtuple]:
        """
        Validate and filter generated sixtuples to ensure musical correctness
        """

        if not generated_sixtuples:
            return generated_sixtuples

        validated_sixtuples = []

        for sixtuple in generated_sixtuples:
            try:
                # Validate each component can be parsed
                bar_num = int(sixtuple.bar.split("_")[1])
                position_num = int(sixtuple.position.split("_")[1])
                pitch_num = int(sixtuple.pitch.split("_")[1])
                duration_val = float(sixtuple.duration.split("_")[1])
                velocity_num = int(sixtuple.velocity.split("_")[1])
                tempo_num = int(sixtuple.tempo.split("_")[1])

                # Filter out potentially problematic values
                if (
                    0 <= bar_num <= MAX_BAR_NUMBER  # Reasonable bar range
                    and 0 <= position_num <= MAX_POSITION_NUMBER  # Valid 16th note positions
                    and MIN_MIDI_PITCH <= pitch_num <= MAX_MIDI_PITCH  # Valid MIDI pitch range
                    and MIN_NOTE_DURATION <= duration_val <= MAX_NOTE_DURATION  # Reasonable note durations
                    and MIN_MIDI_VELOCITY <= velocity_num <= MAX_MIDI_VELOCITY  # Valid MIDI velocity
                    and MIN_TEMPO <= tempo_num <= MAX_TEMPO  # Reasonable tempo range
                ):
                    validated_sixtuples.append(sixtuple)

            except (ValueError, IndexError):
                continue

        return validated_sixtuples

    def generate_music_stream(
        self, seed_sequence: list[tuple[int, int, int, int, int, int]], generation_length: int = GENERATION_LENGTH
    ):
        """
        Generate a complete music stream from a seed sequence, supporting chords.
        """
        generated_chords = self.generate_sequence(seed_sequence, generation_length)
        # Validate each chord, remove empty chords
        validated_chords = [list(self._validate_generated_sequence(chord)) for chord in generated_chords]
        validated_chords = [chord for chord in validated_chords if chord]
        normalized_sixtuples = self._normalize_generated_chords(validated_chords)
        return detokenize(normalized_sixtuples)


def load_seed_sequence_from_midi(midi_path: str, processed_dataset_id: str, sequence_length: int = 8):
    """
    Load a MIDI file, tokenize it, and return the first sequence_length events
    as numeric sixtuples.
    """

    # Load token maps
    token_maps_dir = os.path.join(TOKEN_MAPS_DIR, processed_dataset_id)
    feature_names = ["bar", "position", "pitch", "duration", "velocity", "tempo"]
    token_maps = {}

    for feature_name in feature_names:
        map_path = os.path.join(token_maps_dir, f"{feature_name}_map.json")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Token map not found: {map_path}")
        with open(map_path) as f:
            token_maps[feature_name] = json.load(f)

    # Load and tokenize MIDI file
    stream = converter.parse(midi_path)
    tokenizer = Tokenizer(processed_dataset_id)
    tokenized_sixtuples = tokenizer.tokenize(stream)

    if len(tokenized_sixtuples) < sequence_length:
        raise ValueError(
            f"The MIDI file has only {len(tokenized_sixtuples)} events, at least {sequence_length} required."
        )

    # Take fist sequence_length events and convert to numeric values
    seed_sequence = []
    for sixtuple in tokenized_sixtuples[:sequence_length]:
        numeric_values = []
        tokens = [sixtuple.bar, sixtuple.position, sixtuple.pitch, sixtuple.duration, sixtuple.velocity, sixtuple.tempo]
        for feature_name, token in zip(feature_names, tokens, strict=True):
            idx = token_maps[feature_name][token]
            numeric_values.append(idx)
        seed_sequence.append(tuple(numeric_values))

    return seed_sequence
