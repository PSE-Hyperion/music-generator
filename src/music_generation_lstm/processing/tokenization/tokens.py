from abc import ABC, abstractmethod

import numpy as np


class BaseToken(ABC):
    """
    Abstract base class for token classes.
    """
    __slots__ = ('_features',)

    @abstractmethod
    @property
    def FEATURE_NAMES(self):  # noqa: N802
        pass

    #TODO
    #@abstractmethod
    #@property
    #def FEATURE_REPRS(self):
    #    pass

    def __init__(self, features):
        self._features = np.array(features)

    def __getattr__(self, name):
        if name in self.FEATURE_NAMES:


            idx = self.FEATURE_NAMES.index(name)
            return self._features[idx]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __array__(self):
        return self._features

    def __len__(self):
        return len(self._features)


class REMI6(BaseToken):
    @property
    def FEATURE_NAMES(self):  # noqa: N802
        return [
            'bar',
            'position',
            'pitch',
            'duration',
            'velocity',
            'tempo'
        ]
