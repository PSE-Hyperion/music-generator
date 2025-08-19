from __future__ import annotations

import logging
import os

from music21 import stream

from groove_panda.config import Config

config = Config()
logger = logging.getLogger(__name__)


def generate_sheet_music(
    output_name: str,
    output_directory: str,
    score: stream.Stream,
) -> None:
    """
    Generates MusicXML for the given music21 Stream (score) if sheet music creation is enabled.
    Args:
        output_name: The name of the song to which we are creating sheet music.
        output_directory: The directory of the generated song where we will save the sheet music
        score (stream.Stream): The music21 Stream to write.
    Effects:
        - Writes “<prefix><output_name>.xml” (e.g. “sheet_mysong.xml”), overwriting any existing file.
        - Prints status messages; does not return a value.
    """
    # Only generate sheet music if enabled in config
    if not config.create_sheet_music:
        logger.info("Sheet music generation is disabled (CREATE_SHEET_MUSIC=False).")
        return

    # Construct filename and write, overwriting if necessary
    prefix = "sheet_"
    filename = f"{prefix}{output_name}.xml"
    filepath = os.path.join(output_directory, filename)
    score.write("musicxml", fp=filepath)
    logger.info("Wrote MusicXML to %s", filepath)
