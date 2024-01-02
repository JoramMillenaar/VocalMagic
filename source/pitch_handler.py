from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from pitch_detection.pitch_detectors import PitchDetector
from source.audio_stretchers import StretchAlgorithm
from source.base import AudioProcessor
from source.dataclasses import WaveID
from source.services import snap_nearest_index


class PitchHandler(AudioProcessor, ABC):
    def __init__(self, sample_rate, pitch_detector: PitchDetector, stretch_algorithm: StretchAlgorithm):
        super().__init__(sample_rate)
        self.pitch_detector = pitch_detector
        self.stretch_algorithm = stretch_algorithm

    def process(self, stream_item: np.ndarray) -> np.ndarray:
        stream_item = self.pitch_detector.process(stream_item)
        return self.handle(stream_item, self.pitch_detector.base_frequency)

    @abstractmethod
    def handle(self, audio_chunk: np.ndarray, f0: WaveID) -> np.ndarray:
        pass


class MonoTonePitchHandler(PitchHandler):
    def __init__(self,
                 sample_rate,
                 pitch_detector: PitchDetector,
                 frequency, stretch_algorithm):
        super().__init__(sample_rate, pitch_detector, stretch_algorithm)
        self.frequency = frequency

    def handle(self, audio_chunk: np.ndarray, f0: WaveID):
        stretch_factor = f0.frequency / self.frequency
        return self.stretch_algorithm.stretch(audio_chunk, factor=stretch_factor)


class SelectionPitchHandler(PitchHandler):
    def __init__(self,
                 sample_rate,
                 pitch_detector: PitchDetector,
                 frequency_selection: Sequence[float],
                 stretch_algorithm):
        super().__init__(sample_rate, pitch_detector, stretch_algorithm)
        self.frequency_selection = frequency_selection

    def handle(self, audio_chunk: np.ndarray, f0: WaveID):
        desired_frequency = self.frequency_selection[snap_nearest_index(f0.frequency, self.frequency_selection)]
        stretch_factor = f0.frequency / desired_frequency
        return self.stretch_algorithm.stretch(audio_chunk, factor=stretch_factor)
