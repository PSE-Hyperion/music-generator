import logging
import os

from mido import MidiFile
from music21 import stream

from groove_panda.config import Config

config = Config()
logger = logging.getLogger(__name__)


def write_midi(output_dir: str, result_id: str, result: stream.Stream | MidiFile) -> None:
    """
    Writes given stream into it's own folder in the results dir as a midi file
    """

    logger.info("Started saving %s...", result_id)

    try:
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(result, stream.Stream):
            _write_midi_music21(result_id, output_dir, result)
        else:
            _write_midi_mido(result_id, output_dir, result)
    except Exception as e:
        raise Exception(f"Saving {result_id} was unsuccessful.") from e

    logger.info("Finished saving %s.", result_id)


def _write_midi_music21(result_id: str, result_dir: str, stream: stream.Stream) -> None:
    stream.write("midi", fp=os.path.join(result_dir, f"{result_id}.midi"))


def _write_midi_mido(result_id: str, result_dir: str, midi_file: MidiFile) -> None:
    midi_file.save(os.path.join(result_dir, f"{result_id}.mid"))
