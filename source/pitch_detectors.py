from abc import ABC, abstractmethod

import numpy as np

from source.base import AudioProcessor
from source.fft import FFTAnalyser
from source.services import base_frequency_indexes, loudest_harmonic_of_loudest_base
from source.dataclasses import WaveID
from source.window_managers import AudioOverlapProcessor


class PitchDetector(AudioProcessor, ABC):
    def __init__(self, sample_rate: int):
        super().__init__(sample_rate)
        self._base_frequency = None

    @property
    def base_frequency(self) -> WaveID:
        return self._base_frequency

    @base_frequency.setter
    def base_frequency(self, value: WaveID):
        self._base_frequency = value

    @abstractmethod
    def process(self, audio_chunk: np.array) -> np.array:
        pass


class SimplePitchDetector(PitchDetector):
    def __init__(self, sample_rate: int, frequency_resolution: int = 3):
        super().__init__(sample_rate)
        self.fft = FFTAnalyser(sample_rate)
        self.overlap = AudioOverlapProcessor(sample_rate=sample_rate, frequency_resolution=frequency_resolution)

    def process(self, audio_chunk: np.array) -> np.array:
        original_chunk_size = len(audio_chunk)
        audio_chunk = self.overlap.process(audio_chunk)
        fft_analytics = self.fft.analyse(audio_chunk)
        index_loudest = np.argmax(fft_analytics.magnitudes)
        frequency = fft_analytics.frequency_range[index_loudest]
        amplitude = fft_analytics.magnitudes[index_loudest] / fft_analytics.chunk_size
        self.base_frequency = WaveID(frequency, amplitude)
        return audio_chunk[-original_chunk_size:]


class HarmonicPitchDetector(PitchDetector):
    def __init__(self, sample_rate, lenience: float = 1, frequency_resolution: int = 3):
        super().__init__(sample_rate)
        self.lenience = lenience
        self._cached_indexes = None
        self.fft = FFTAnalyser(sample_rate)
        self.overlap = AudioOverlapProcessor(sample_rate=sample_rate, frequency_resolution=frequency_resolution)

    def harmonic_indexes(self, frequencies):
        if self._cached_indexes is None:
            self._cached_indexes = base_frequency_indexes(frequencies, self.lenience)
        return self._cached_indexes

    def process(self, audio_chunk):
        original_chunk_size = len(audio_chunk)
        audio_chunk = self.overlap.process(audio_chunk)
        fft_analytics = self.fft.analyse(audio_chunk)
        indexes = self.harmonic_indexes(fft_analytics.frequency_range)
        i = loudest_harmonic_of_loudest_base(indexes, fft_analytics.magnitudes)
        f = fft_analytics.frequency_range[i]
        a = fft_analytics.magnitudes[i] / fft_analytics.chunk_size
        self.base_frequency = WaveID(f, a)
        return audio_chunk[-original_chunk_size:]
