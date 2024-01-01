from abc import ABC, abstractmethod

import numpy as np

from source.base import AudioProcessor
from pitch_detection.fft import FFTAnalyser
from pitch_detection.yin import yin_pitch_detection
from source.services import base_frequency_indexes, loudest_harmonic_of_loudest_base
from source.dataclasses import WaveID


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

    def process(self, audio_chunk: np.array) -> np.array:
        self.base_frequency = self.extract_base_frequency(audio_chunk)
        return audio_chunk

    @abstractmethod
    def extract_base_frequency(self, audio_chunk: np.array) -> WaveID:
        pass


class SimplePitchDetector(PitchDetector):
    def __init__(self, sample_rate: int, frequency_resolution: int = 3):
        super().__init__(sample_rate)
        self.fft = FFTAnalyser(sample_rate, frequency_resolution)

    def extract_base_frequency(self, audio_chunk: np.array) -> np.array:
        fft_analytics = self.fft.analyse(audio_chunk)
        index_loudest = np.argmax(fft_analytics.magnitudes)
        frequency = fft_analytics.frequency_range[index_loudest]
        amplitude = fft_analytics.magnitudes[index_loudest] / fft_analytics.chunk_size
        return WaveID(frequency, amplitude)


class HarmonicPitchDetector(PitchDetector):
    def __init__(self, sample_rate, lenience: float = 1, frequency_resolution: int = 3):
        super().__init__(sample_rate)
        self.lenience = lenience
        self._cached_indexes = None
        self.fft = FFTAnalyser(sample_rate, frequency_resolution)

    def harmonic_indexes(self, frequencies):
        if self._cached_indexes is None:
            self._cached_indexes = base_frequency_indexes(frequencies, self.lenience)
        return self._cached_indexes

    def extract_base_frequency(self, audio_chunk):
        fft_analytics = self.fft.analyse(audio_chunk)
        indexes = self.harmonic_indexes(fft_analytics.frequency_range)
        index = loudest_harmonic_of_loudest_base(indexes, fft_analytics.magnitudes)
        f = fft_analytics.frequency_range[index]
        a = fft_analytics.magnitudes[index] / fft_analytics.chunk_size
        return WaveID(f, a)


class YinPitchDetector(PitchDetector):
    def __init__(self, sample_rate, threshold=0.1):
        super().__init__(sample_rate)
        self.threshold = threshold

    def extract_base_frequency(self, audio_chunk) -> WaveID:
        frequency = yin_pitch_detection(audio_chunk, self.sample_rate, self.threshold)
        return WaveID(frequency or 0, 1.0)