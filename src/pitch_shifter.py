from abc import ABC

import numpy as np

from pitch_detection.pitch_detectors import YinPitchDetector
from src.base import AudioProcessor
from src.frequency_getters import FrequencyGetter
from src.services import resample_to_size


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
        return resample_to_size(audio_chunk, stretch_factor)
