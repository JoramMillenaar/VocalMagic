from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from source.services import resample_audio, snap_nearest_index
from source.dataclasses import AudioAnalytics, WaveID


class Manipulator(ABC):
    @abstractmethod
    def manipulate(self, audio_chunk: np.ndarray, analytics: AudioAnalytics, wave_id: WaveID):
        pass


class MonoToneManipulator(Manipulator):
    def __init__(self, frequency: float):
        self.frequency = frequency

    def manipulate(self, audio_chunk: np.ndarray, analytics: AudioAnalytics, wave_id: WaveID):
        stretch_factor = wave_id.frequency / self.frequency
        return resample_audio(audio_chunk, chunk_size=len(audio_chunk), factor=stretch_factor)


class AutoTuneManipulator(Manipulator):
    def __init__(self, frequencies: Sequence[float]):
        self.frequencies = frequencies

    def manipulate(self, audio_chunk: np.ndarray, analytics: AudioAnalytics, wave_id: WaveID):
        desired_frequency = self.frequencies[snap_nearest_index(wave_id.frequency, self.frequencies)]
        stretch_factor = wave_id.frequency / desired_frequency
        return resample_audio(audio_chunk, chunk_size=len(audio_chunk), factor=stretch_factor)
