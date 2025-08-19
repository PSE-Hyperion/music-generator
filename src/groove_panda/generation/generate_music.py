import logging
import os

from groove_panda.config import Config
from groove_panda.generation.generate import MusicGenerator
from groove_panda.midi import writer
from groove_panda.midi.parser import parse_midi
from groove_panda.models.model_io import load_model
from groove_panda.processing.tokenization import token_map_io
from groove_panda.processing.tokenization.tokenizer import Tokenizer, detokenize
from groove_panda.utils import overwrite_json

config = Config()
logger = logging.getLogger(__name__)


def generate_music(model_name: str, input_name: str, output_name: str):
    """
    Load model -> Load input MIDI -> Generate -> Save output
    """

    logger.info(f"Starting music generation with model: {model_name}")

    # Load the trained model
    model, model_metadata = load_model(model_name)
    processed_dataset_id = model_metadata.get("processed_dataset_id")
    if not processed_dataset_id:
        raise ValueError(
            "The loaded model metadata does not contain a 'processed_dataset_id'. "
            "Please ensure the model was saved with the correct dataset reference."
        )

    token_maps, metadata, reverse_mappings = token_map_io.load_token_maps(processed_dataset_id)
    generator = MusicGenerator(model.model, token_maps, reverse_mappings, metadata)

    # Load seed sequence from input MIDI file
    input_midi_path = None
    for ext in config.allowed_music_file_extensions:
        candidate = os.path.join(config.input_midi_dir, f"{input_name}{ext}")
        if os.path.exists(candidate):
            input_midi_path = candidate
            break
    if input_midi_path is None:
        raise FileNotFoundError(f"Input MIDI file not found: {input_name}")

    logger.info(f"Loading seed sequence from: {input_midi_path}")

    score = parse_midi(input_midi_path)
    tokenizer = Tokenizer(processed_dataset_id)
    sixtuples = tokenizer.tokenize_original_key(score)

    # Convert to numeric tuples
    seed_sequence = []
    seed_sixtuple = []
    for sixtuple in sixtuples[: generator.sequence_length]:
        numeric_tuple = (
            token_maps["bar"][sixtuple.bar],
            token_maps["position"][sixtuple.position],
            token_maps["pitch"][sixtuple.pitch],
            token_maps["duration"][sixtuple.duration],
            token_maps["velocity"][sixtuple.velocity],
            token_maps["tempo"][sixtuple.tempo],
        )
        seed_sequence.append(numeric_tuple)
        seed_sixtuple.append(sixtuple)

    # Generate and save
    generated_sixtuples = generator.generate_sequence(seed_sequence)

    # Save the song with the input sequence
    generated_stream_full = detokenize(seed_sixtuple + generated_sixtuples)
    writer.write_midi(output_name, generated_stream_full)

    # Also save just the generation
    generated_stream_cont = detokenize(generated_sixtuples)
    writer.write_midi(output_name + "_cont", generated_stream_cont)

    logger.info(f"Music generation completed! Output saved as: {output_name}")

    # Save token tuples as JSON, only generate sheet music if enabled in config
    if not config.save_token_json:
        logger.info("Saving Token as json is disabled (SAVE_TOKEN_JSON=False).")
        return
    token_tuples = [
        {
            "bar": tuple.bar,
            "position": tuple.position,
            "pitch": tuple.pitch,
            "duration": tuple.duration,
            "velocity": tuple.velocity,
            "tempo": tuple.tempo,
        }
        for tuple in generated_sixtuples
    ]
    base_name = os.path.splitext(os.path.basename(output_name))[0]
    json_output_path = os.path.join(config.result_tokens_dir, f"{base_name}.json")

    # Save the token tuples
    os.makedirs(config.result_tokens_dir, exist_ok=True)
    overwrite_json(json_output_path, token_tuples)

    logger.info("Saved Tokens as json")
