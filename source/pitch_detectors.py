from abc import ABC, abstractmethod

import numpy as np

from source.fft import FFTAnalyser
from source.services import base_frequency_indexes, loudest_harmonic_of_loudest_base
from source.dataclasses import WaveID


class PitchDetector(ABC):
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    @abstractmethod
    def extract_wave_id(self, audio_chunk: np.array) -> WaveID:
        pass


class SimplePitchDetector(PitchDetector):
    def __init__(self, sample_rate: int):
        super().__init__(sample_rate)
        self.analyser = FFTAnalyser(sample_rate)
        self.primed = False

    def extract_wave_id(self, audio_chunk) -> WaveID:
        if not self.primed:
            self.analyser.prime(len(audio_chunk))
        analytics = self.analyser.analyse(audio_chunk)
        index_loudest = np.argmax(analytics.magnitudes)
        frequency = analytics.frequency_range[index_loudest]
        amplitude = analytics.magnitudes[index_loudest] / analytics.chunk_size
        return WaveID(frequency, amplitude)


class HarmonicPitchDetector(PitchDetector):
    def __init__(self, sample_rate, lenience: float = 1):
        super().__init__(sample_rate)
        self.lenience = lenience
        self._cached_indexes = None
        self.analyser = FFTAnalyser(sample_rate)

    def harmonic_indexes(self, frequencies):
        if self._cached_indexes is None:
            self._cached_indexes = base_frequency_indexes(frequencies, self.lenience)
        return self._cached_indexes

    def extract_wave_id(self, audio_chunk):
        analytics = self.analyser.analyse(audio_chunk)
        indexes = self.harmonic_indexes(analytics.frequency_range)
        i = loudest_harmonic_of_loudest_base(indexes, analytics.magnitudes)
        r = WaveID(analytics.frequency_range[i], analytics.magnitudes[i] / analytics.chunk_size)
        return r
