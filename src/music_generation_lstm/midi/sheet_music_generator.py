from __future__ import annotations

from itertools import count
import logging
import os

from music21 import stream

from music_generation_lstm.config import CREATE_SHEET_MUSIC, OUTPUT_SHEET_MUSIC_DIR

logger = logging.getLogger(__name__)

# Counter for automatic, sequential filenames
SHEET_INDEX = count(0)


def generate_sheet_music(
    score: stream.Stream,
    prefix: str = "score_",
) -> None:
    """
    Generates MusicXML for the given music21 Stream (score) if sheet music creation is enabled.
    Args:
        score (stream.Stream): The music21 Stream to write.
        prefix (str): Filename prefix ("score_").
    Side effects:
        - Ensures OUTPUT_SHEET_MUSIC_DIR exists.
        - Writes “<prefix><index>.xml” (e.g. “score_3.xml”), overwriting any existing file.
        - Prints status messages; does not return a value.
    """
    # Only generate sheet music if enabled in config
    if not CREATE_SHEET_MUSIC:
        logger.info("Sheet music generation is disabled (CREATE_SHEET_MUSIC=False).")
        return

    os.makedirs(OUTPUT_SHEET_MUSIC_DIR, exist_ok=True)

    # Construct filename and write, overwriting if necessary
    index = next(SHEET_INDEX)
    filename = f"{prefix}{index}.xml"
    filepath = os.path.join(OUTPUT_SHEET_MUSIC_DIR, filename)
    score.write("musicxml", fp=filepath)
    logger.info("Wrote MusicXML to %s", filepath)
