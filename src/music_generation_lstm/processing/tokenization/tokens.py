from dataclasses import dataclass

from numpy import array, uint8

from music_generation_lstm.midi.features import represent_feature


class BaseToken:
    """
    Abstract base class for token classes.
    """
    __slots__ = ()
    def __array__(self):
        return array([self.__getattribute__(s) for s in self.__slots__])

    def __len__(self):
        return len(self.__slots__)

    def __str__(self):
        feature_reprs = []
        for s in self.__slots__:
            name, value = s, self.__getattribute__(s)
            feature_reprs.append('_')
            feature_reprs.append(represent_feature(name, value))
        return ''.join(feature_reprs).removeprefix('_')

@dataclass(slots=True)
class REMI6(BaseToken):
    bar: uint8
    position: uint8
    pitch: uint8
    duration: uint8
    velocity: uint8
    tempo: uint8

