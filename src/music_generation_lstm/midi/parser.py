from collections.abc import Iterable
import logging
from pathlib import Path

from music21 import converter, stream

logger = logging.getLogger(__name__)

def m21_parse_midi_batch(midi_paths: Iterable[Path]) -> list[stream.Score]:
    scores = []
    idx = -1

    for idx, p in enumerate(midi_paths):
        logger.debug("Parsing item %s: %s", idx + 1, p.name)
        s = m21_parse_midi_item(p)
        if s is not None:
            scores.append(s)

    return scores


def m21_parse_midi_item(midi_path: Path) -> stream:
    try:
        parsed = converter.parseFile(midi_path, format="midi")
        return _normalize_to_score(parsed)
    except:
        logger.warning("Parsing of file %s failed, skipping", midi_path)


def _normalize_to_score(stream_object: stream):
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

