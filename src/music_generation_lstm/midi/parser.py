from collections.abc import Iterable
import logging
from pathlib import Path

from music21 import converter, stream

from music_generation_lstm.step import pipeline_step

"""
This class is used for conversion of MIDI files into parsable formats for tokenization or analysis.
We're using m21 only, but other implementations could be added here.
"""

logger = logging.getLogger(__name__)


@pipeline_step(kind='parser')
def m21_parse_midi_batch(midi_paths: Iterable[Path]) -> list[stream.Score]:
    """
    Batch execution of the method below (filtering None objects).
    Used for the processing pipeline.
    Can work with any iterable argument, lazy (generator) or eager (list).
    """
    scores = []
    idx = -1

    for idx, p in enumerate(midi_paths):
        logger.debug("Parsing item %s: %s", idx + 1, p.name)
        s = m21_parse_midi(p)
        if s is not None:
            scores.append(s)

    return scores


def m21_parse_midi(midi_path: Path) -> stream:
    """
    Converts a single midi file (by it's file path) into a music21 stream.score which is pretty handy.
    It's pretty slow though.
    """
    try:
        parsed = converter.parseFile(midi_path, format="midi")
        return _normalize_to_score(parsed)
    except Exception as e:
        logger.warning("Parsing of file %s failed (%s), skipping", midi_path, e)


def _normalize_to_score(stream_object: stream):
    """
    Not every midi file is parsed into a m21 stream.score.
    It can contain a single part (parsed into stream.Part) or a bunch of scores (stream.Opus).
    This method normalizes them all to scores. It either builds a score with only one part
    or takes the first score of an opus.
    """
    if isinstance(stream_object, stream.Score):
        score = stream_object
        logger.debug("Parsed directly to score")
    elif isinstance(stream_object, stream.Part):
        score = stream.Score()
        score.insert(0, stream_object)
        if stream_object.metadata:
            score.metadata = stream_object.metadata
        logger.debug("Parsed to part, built score with it")
    elif isinstance(stream_object, stream.Opus):
        score = stream_object.scores.first()
        logger.debug("Parsed to opus, took only first score of it")
    else:
        score = None

    return score

