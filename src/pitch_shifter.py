from numpy._typing import NDArray
from pypitch.detectors import PitchDetector

from src.frequency_getters import FrequencyGetter
from src.services import resample_to_size


class PitchShifter:
    def __init__(self, frequency_getter: FrequencyGetter, pitch_detector: PitchDetector):
        self.frequency_getter = frequency_getter
        self.pitch_detector = pitch_detector

    def process(self, audio_chunk: NDArray) -> NDArray:
        frequency = self.pitch_detector.detect_frequency(audio_chunk)
        target_frequency = self.frequency_getter.get_target_frequency(frequency)
        return resample_to_size(audio_chunk, factor=frequency / target_frequency)
