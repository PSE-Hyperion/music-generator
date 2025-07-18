from dataclasses import dataclass

import numpy as np
from tensorflow.keras.models import Model  # type: ignore

from groove_panda.config import GENERATION_LENGTH
from groove_panda.processing.process import sequence_to_model_input
from groove_panda.processing.tokenization.tokenizer import Sixtuple


@dataclass
class ModelPredictions:
    bar: np.ndarray
    position: np.ndarray
    pitch: np.ndarray
    duration: np.ndarray
    velocity: np.ndarray
    tempo: np.ndarray


class MusicGenerator:
    def __init__(
        self, model: Model, token_maps: dict, reverse_mappings: dict, metadata: dict, temperature: float = 1.0
    ):
        self.model = model
        self.token_maps = token_maps
        self.reverse_mappings = reverse_mappings
        self.metadata = metadata
        self.temperature = temperature

        first_input = model.inputs[0]
        self.sequence_length = first_input.shape[1]

    def _apply_temperature_sampling(self, probabilities: np.ndarray, temperature: float) -> int:
        """
        Apply temperature sampling to the predicted probabilities
        """

        if temperature == 0:
            # Greedy sampling (deterministic)
            return int(np.argmax(probabilities))

        # Apply temperature scaling
        scaled_logits = np.log(probabilities + 1e-8) / temperature

        # Apply softmax to get new probabilities
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        new_probabilities = exp_logits / np.sum(exp_logits)

        # Sample from the new distribution
        return np.random.choice(len(new_probabilities), p=new_probabilities)

    def _predictions_to_sixtuple(self, predictions: ModelPredictions) -> Sixtuple:
        """
        Convert model predictions to a Sixtuple using temperature sampling
        """

        # Apply temperature sampling to each feature prediction
        bar_idx = self._apply_temperature_sampling(predictions.bar[0], self.temperature)
        position_idx = self._apply_temperature_sampling(predictions.position[0], self.temperature)
        pitch_idx = self._apply_temperature_sampling(predictions.pitch[0], self.temperature)
        duration_idx = self._apply_temperature_sampling(predictions.duration[0], self.temperature)
        velocity_idx = self._apply_temperature_sampling(predictions.velocity[0], self.temperature)
        tempo_idx = self._apply_temperature_sampling(predictions.tempo[0], self.temperature)

        # Convert indices back to tokens using reverse mappings
        tokens = [
            self.reverse_mappings["bar"][bar_idx],
            self.reverse_mappings["position"][position_idx],
            self.reverse_mappings["pitch"][pitch_idx],
            self.reverse_mappings["duration"][duration_idx],
            self.reverse_mappings["velocity"][velocity_idx],
            self.reverse_mappings["tempo"][tempo_idx],
        ]

        return Sixtuple(
            bar=tokens[0].split("_")[1],
            position=tokens[1].split("_")[1],
            pitch=tokens[2].split("_")[1],
            duration=tokens[3].split("_")[1],
            velocity=tokens[4].split("_")[1],
            tempo=tokens[5].split("_")[1],
        )

    def _sixtuple_to_numeric_tuple(self, sixtuple: Sixtuple) -> tuple[int, int, int, int, int, int]:
        return (
            self.token_maps["bar"][sixtuple.bar],
            self.token_maps["position"][sixtuple.position],
            self.token_maps["pitch"][sixtuple.pitch],
            self.token_maps["duration"][sixtuple.duration],
            self.token_maps["velocity"][sixtuple.velocity],
            self.token_maps["tempo"][sixtuple.tempo],
        )

    def generate_sequence(
        self, seed_sequence: list[tuple[int, int, int, int, int, int]], generation_length: int = GENERATION_LENGTH
    ) -> list[Sixtuple]:
        """
        Generate a sequence of music events using the trained model.
        """
        print(f"Starting generation with temperature: {self.temperature}")
        print(f"Generation length: {generation_length}")

        if len(seed_sequence) != self.sequence_length:
            raise ValueError(f"Seed sequence must be exactly {self.sequence_length} events long")

        current_sequence = seed_sequence
        generated_events = []

        for i in range(generation_length):
            model_input = sequence_to_model_input(current_sequence)
            predictions = self.model.predict(model_input, verbose=0)

            predictions = ModelPredictions(
                bar=predictions[0],
                position=predictions[1],
                pitch=predictions[2],
                duration=predictions[3],
                velocity=predictions[4],
                tempo=predictions[5],
            )

            sixtuple = self._predictions_to_sixtuple(predictions)
            generated_events.append(sixtuple)

            # Update sequence für nächste Vorhersage
            numeric_event = self._sixtuple_to_numeric_tuple(sixtuple)
            current_sequence = [*current_sequence[1:], numeric_event]

            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{generation_length} events")

        print("Generation completed")
        return generated_events
