from abc import ABC, abstractmethod

import numpy as np

from source.services import base_frequency_indexes, loudest_harmonic_of_loudest_base
from source.dataclasses import WaveID, AudioAnalytics


class FrequencyExtractor(ABC):
    @abstractmethod
    def extract_wave_id(self, analytics: AudioAnalytics) -> WaveID:
        pass


class SimpleFrequencyExtractor(FrequencyExtractor):
    def extract_wave_id(self, analytics) -> WaveID:
        index_loudest = np.argmax(analytics.magnitudes)
        frequency = analytics.frequency_range[index_loudest]
        amplitude = analytics.magnitudes[index_loudest] / analytics.chunk_size
        return WaveID(frequency, amplitude)


class HarmonicFrequencyExtractor(FrequencyExtractor):
    def __init__(self, lenience: float = 1):
        self.lenience = lenience
        self._cached_indexes = None

    def harmonic_indexes(self, frequencies):
        if self._cached_indexes is None:
            self._cached_indexes = base_frequency_indexes(frequencies, self.lenience)
        return self._cached_indexes

    def extract_wave_id(self, analytics: AudioAnalytics):
        indexes = self.harmonic_indexes(analytics.frequency_range)
        i = loudest_harmonic_of_loudest_base(indexes, analytics.magnitudes)
        r = WaveID(analytics.frequency_range[i], analytics.magnitudes[i] / analytics.chunk_size)
        return r
