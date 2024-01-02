from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import resample


class StretchAlgorithm(ABC):
    @abstractmethod
    def stretch(self, audio_chunk: np.ndarray, factor: float) -> np.ndarray:
        pass


class ResampleStretchAlgorithm(StretchAlgorithm):
    def stretch(self, audio_chunk: np.ndarray, factor: float) -> np.ndarray:
        return resample(audio_chunk, num=int(len(audio_chunk) * factor))
