from music21.pitch import Pitch


def represent_feature(name: str, value: int) -> str:
    return _FEATURE_PREFIXES.get(name, '') + _FEATURE_REPRS.get(name, (lambda x: str(x)))(value)


_FEATURE_PREFIXES = {
    'bar': 'BAR_',
    'position': 'POS_',
    'pitch': 'PITCH_',
    'duration': 'DUR_',
    'velocity': 'VELO_',
    'tempo': 'TEMPO_',
}

_FEATURE_REPRS = {
    'pitch': (lambda x: str(Pitch(x)))
}

