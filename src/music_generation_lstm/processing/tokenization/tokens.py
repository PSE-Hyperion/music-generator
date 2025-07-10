from abc import ABC, abstractmethod
import collections
from dataclasses import dataclass

from numpy import array, uint8

from music_generation_lstm.midi.features import represent_feature


@dataclass(slots=True)
class BaseToken(ABC):  # noqa: B024
    """
    Abstract base class for token classes.
    Token class should have the decorator @dataclass(slots=true) and should be a list of used features.
    This will generate constructors and make them accessible with <token>.<feature> .
    """
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
class Tuple(BaseToken, ABC):
    """
    Should be regarded as a base for all note-as-tuple-like tokens.
    Every note of a song is exactly one token. The NTuple has an arbitrary number of features of a note.
    pitch: 8 bit int with the MIDI-code for the pitch of a note.
    """
    @property
    @abstractmethod
    # Placeholder for field to implement
    def pitch(self):
        pass

@dataclass(slots=True)
class HexTuple(BaseToken):
    """
    Implementation of the NTuple with 6 selected features:
    pitch: 8 bit int, the MIDI-code for the pitch of a note
    bar: 8 bit int, the bar of the song where the note starts
    position: 8 bit int, position of the start of the note within the ar
    duration: 8 bit int, duration of the note as a count of steps (dependent of the chosen quantization)
    velocity: 8 bit int, the velocity of the note
    tempo: 8 bit int, the current tempo in the track where the note is positioned
    """
    pitch: uint8
    bar: uint8
    position: uint8
    duration: uint8
    velocity: uint8
    tempo: uint8


class Vocabulary:
    def __init__(self, token_type: type(BaseToken)):
        self.token_type = token_type
        self.__slots__ = token_type.__slots__
        for s in self.__slots__:
            self.__setattr__(s, collections.Counter(s))

    def update(self, token: BaseToken):
        if not isinstance(token, self.token_type):
            raise TypeError(f"Expected {self.token_type} but got {type(token)}")
        for s in self.__slots__:
            self.__getattribute__(s).update(token.__getattribute__(s))

    def generate_encoding_maps(self):
        combined_map = collections.ChainMap()
        for s in self.__slots__:
            counter: collections.Counter = self.__getattribute__(s)
            ordered_list = counter.most_common()
            ordered_list.reverse()
            map = {}
            for i, key in enumerate(ordered_list):
                map.update([key, i])
            combined_map.new_child(map)
        return combined_map


