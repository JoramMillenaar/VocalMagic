from abc import ABC
from functools import cached_property

import numpy as np

from source.dataclasses import AudioAnalytics


class AudioAnalyserBase(ABC):
    def __init__(self, chunk_size: int, sample_rate: int):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate

    @cached_property
    def frequency_range(self):
        return np.fft.rfftfreq(self.chunk_size, d=1 / self.sample_rate)

    def analyse(self, audio_chunk: np.array) -> AudioAnalytics:
        complex_spectrum = np.fft.rfft(audio_chunk)
        return AudioAnalytics(
            chunk_size=self.chunk_size,
            sample_rate=self.sample_rate,
            frequency_range=self.frequency_range,
            complex_spectrum=complex_spectrum,
            magnitudes=np.abs(complex_spectrum)
        )


class AudioAnalyser(AudioAnalyserBase):
    pass


class RangedAudioAnalyser(AudioAnalyserBase):
    def __init__(self, chunk_size: int, sample_rate: int, min_freq: float = 85, max_freq: float = 1100):
        super().__init__(chunk_size, sample_rate)
        self.min_freq = min_freq
        self.max_freq = max_freq

    @cached_property
    def min_index(self):
        return np.searchsorted(self.frequency_range, self.min_freq)

    @cached_property
    def max_index(self):
        return np.searchsorted(self.frequency_range, self.max_freq, side='right')

    @cached_property
    def ranged_frequency_range(self):
        return self.frequency_range[self.min_index:self.max_index]

    def analyse(self, audio_chunk: np.array) -> AudioAnalytics:
        complex_spectrum = np.fft.rfft(audio_chunk)[self.min_index:self.max_index]
        return AudioAnalytics(
            chunk_size=self.chunk_size,
            sample_rate=self.sample_rate,
            frequency_range=self.ranged_frequency_range,
            complex_spectrum=complex_spectrum,
            magnitudes=np.abs(complex_spectrum)
        )
