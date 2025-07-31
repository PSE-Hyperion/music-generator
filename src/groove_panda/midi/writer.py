import logging
import os

from mido import MidiFile
from music21 import stream

from groove_panda.config import Config

config = Config()
logger = logging.getLogger(__name__)


def write_midi(result_id: str, result: stream.Stream | MidiFile):
    #   Writes given stream into it's own folder in the results dir as a midi file
    #   Throws exception, if dir already exists (Could be changed to handle overwriting)
    #

    logger.info("Started saving %s...", result_id)

    try:
        result_dir = os.path.join(config.results_midi_dir, result_id)
        os.makedirs(result_dir, exist_ok=False)

        if isinstance(result, stream.Stream):
            _write_midi_music21(result_id, result_dir, result)
        else:
            _write_midi_mido(result_id, result_dir, result)
    except Exception as e:
        raise Exception(f"Saving {result_id} was unsuccessful.") from e

    logger.info("Finished saving %s.", result_id)


def _write_midi_music21(result_id: str, result_dir: str, stream: stream.Stream):
    stream.write("midi", fp=os.path.join(result_dir, f"{result_id}.midi"))

    # data_managment.add_result_id(result_id) unused


def _write_midi_mido(result_id: str, result_dir: str, midi_file: MidiFile):
    midi_file.save(os.path.join(result_dir, f"{result_id}.mid"))

    # data_managment.add_result_id(result_id) unused
