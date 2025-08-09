from dataclasses import dataclass
import logging

import numpy as np
from tensorflow.keras.models import Model  # type: ignore

from groove_panda.config import Config
from groove_panda.processing.process import sequence_to_model_input
from groove_panda.processing.tokenization.tokenizer import Sixtuple

config = Config()
logger = logging.getLogger(__name__)


@dataclass
class ModelPredictions:
    bar: np.ndarray
    position: np.ndarray
    pitch: np.ndarray
    duration: np.ndarray
    velocity: np.ndarray
    tempo: np.ndarray


class MusicGenerator:
    def __init__(self, model: Model, token_maps: dict, reverse_mappings: dict, metadata: dict):
        self.model = model
        self.token_maps = token_maps
        self.reverse_mappings = reverse_mappings
        self.metadata = metadata
        self.feature_temperatures = config.feature_temperatures

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

        sampled_indices = {}
        # Use config.feature_names
        for feature in config.feature_names:
            feature_predictions = getattr(predictions, feature)[0]
            temp = self.feature_temperatures[feature]

            if temp == 0.0:
                sampled_indices[feature] = np.argmax(feature_predictions)
            else:
                sampled_indices[feature] = self._apply_temperature_sampling(feature_predictions, temp)

        # Convert indices back to tokens using reverse mappings
        tokens = {feature: self.reverse_mappings[feature][sampled_indices[feature]] for feature in config.feature_names}

        return Sixtuple(
            bar=tokens["bar"].split("_")[1],
            position=tokens["position"].split("_")[1],
            pitch=tokens["pitch"].split("_")[1],
            duration=tokens["duration"].split("_")[1],
            velocity=tokens["velocity"].split("_")[1],
            tempo=tokens["tempo"].split("_")[1],
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

    def generate_sequence(self, seed_sequence: list[tuple[int, int, int, int, int, int]]) -> list[Sixtuple]:
        """
        Generate a sequence of music events using the trained model.
        """

        generation_length = config.generation_length

        logger.info(f"Starting generation with feature temperatures: {self.feature_temperatures}")
        logger.info(f"Generation length: {generation_length}")

        if len(seed_sequence) != self.sequence_length:
            raise ValueError(f"Seed sequence must be exactly {self.sequence_length} events long")

        current_sequence = seed_sequence
        generated_events = []

        for i in range(generation_length):
            model_input = sequence_to_model_input(current_sequence)
            predictions = self.model.predict(model_input, verbose=0)
            # Output structure has been changed to a dict.
            # Previously built models that use lists are still supported. Here it checks for the type`.
            if isinstance(predictions, dict):
                predictions = ModelPredictions(
                    bar=predictions["output_bar"],
                    position=predictions["output_position"],
                    pitch=predictions["output_pitch"],
                    duration=predictions["output_duration"],
                    velocity=predictions["output_velocity"],
                    tempo=predictions["output_tempo"],
                )
            else:
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
                logger.info(f"Generated {i + 1}/{generation_length} events")

        logger.info("Generation completed")
        return generated_events
