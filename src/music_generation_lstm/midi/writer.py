import os
import logging

from music21.stream import Stream

from music_generation_lstm.config import RESULTS_MIDI_DIR

logger = logging.getLogger(__name__)


def write_midi(result_id: str, stream: Stream):
    #   Writes given stream into it's own folder in the results dir as a midi file
    #   Throws exception, if dir already exists (Could be changed to handle overwriting)
    #

    logger.info("Started saving %s...", result_id)
    write_dir = os.path.join(RESULTS_MIDI_DIR, result_id)
    os.makedirs(write_dir, exist_ok=False)
    stream.write("midi", fp=os.path.join(write_dir, f"{result_id}.midi"))
    logger.info("Finished saving %s.", result_id)
