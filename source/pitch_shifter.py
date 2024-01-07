from abc import ABC
from typing import Sequence

import numpy as np
from scipy.signal import resample

from pitch_detection.pitch_detectors import YinPitchDetector
from source.base import AudioProcessor
from source.frequency_getters import FrequencyGetter


class PitchShifter(AudioProcessor, ABC):
    def __init__(self, frequency_getter: FrequencyGetter):
        self.frequency_getter = frequency_getter


class YinPitchShifter(PitchShifter):
    def __init__(self, frequency_getter: FrequencyGetter, sample_rate: int):
        super().__init__(frequency_getter)
        self.pitch_detector = YinPitchDetector(sample_rate)

    def process(self, audio_chunk: np.ndarray):
        f0 = self.pitch_detector.extract_base_frequency(audio_chunk)
        target_frequency = self.frequency_getter.get_target_frequency(f0.frequency)
        stretch_factor = f0.frequency / target_frequency
        return resample(audio_chunk, num=int(len(audio_chunk) * stretch_factor))

