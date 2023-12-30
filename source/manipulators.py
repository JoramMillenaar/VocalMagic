from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import rfft, rfftfreq
from scipy.signal import butter, lfilter

from source.dataclasses import WaveID
from source.services import resample_audio, snap_nearest_index


class Manipulator(ABC):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    @abstractmethod
    def manipulate(self, audio_chunk: np.ndarray, wave_id: WaveID):
        pass


class MonoToneManipulator(Manipulator):
    def __init__(self, sample_rate, frequency: float):
        super().__init__(sample_rate)
        self.frequency = frequency

    def manipulate(self, audio_chunk: np.ndarray, wave_id: WaveID):
        stretch_factor = wave_id.frequency / self.frequency
        return resample_audio(audio_chunk, chunk_size=len(audio_chunk), factor=stretch_factor)


class AutoTuneManipulator(Manipulator):
    def __init__(self, sample_rate, frequencies: Sequence[float]):
        super().__init__(sample_rate)
        self.frequencies = frequencies

    def manipulate(self, audio_chunk: np.ndarray, wave_id: WaveID):
        desired_frequency = self.frequencies[snap_nearest_index(wave_id.frequency, self.frequencies)]
        stretch_factor = wave_id.frequency / desired_frequency
        return resample_audio(audio_chunk, chunk_size=len(audio_chunk), factor=stretch_factor)
