from dataclasses import dataclass
from functools import cached_property

import numpy as np

from source.window_managers import AudioOverlapProcessor


@dataclass
class FFTAnalytics:
    chunk_size: int
    sample_rate: int
    frequency_range: np.ndarray
    complex_spectrum: np.ndarray
    magnitudes: np.ndarray


class FFTAnalyser:
    def __init__(self, sample_rate: int, frequency_resolution: int = 3, min_freq: float = 85, max_freq: float = 1100):
        self.sample_rate = sample_rate
        self.frequency_resolution = frequency_resolution
        self.min_freq = min_freq
        self.max_freq = max_freq

        self.overlap = AudioOverlapProcessor(sample_rate=sample_rate, frequency_resolution=frequency_resolution)
        self._frequency_range = None

    def prime(self, chunk_size):
        self._frequency_range = np.fft.rfftfreq(chunk_size, d=1 / self.sample_rate)
        self._frequency_range = self._frequency_range[self.min_index:self.max_index]

    @cached_property
    def min_index(self):
        return np.searchsorted(self.frequency_range, self.min_freq)

    @cached_property
    def max_index(self):
        return np.searchsorted(self.frequency_range, self.max_freq, side='right')

    @property
    def frequency_range(self):
        return self._frequency_range

    def analyse(self, audio_chunk: np.array) -> FFTAnalytics:
        audio_chunk = self.overlap.process(audio_chunk)
        if self._frequency_range is None:
            self.prime(len(audio_chunk))
        complex_spectrum = np.fft.rfft(audio_chunk)[self.min_index:self.max_index]
        return FFTAnalytics(
            chunk_size=len(audio_chunk),
            sample_rate=self.sample_rate,
            frequency_range=self.frequency_range,
            complex_spectrum=complex_spectrum,
            magnitudes=np.abs(complex_spectrum)
        )
