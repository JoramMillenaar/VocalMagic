from abc import ABC, abstractmethod
from typing import Sequence

from source.services import snap_nearest_index


class FrequencyGetter(ABC):
    @abstractmethod
    def get_target_frequency(self, current_frequency: float) -> float:
        pass


class FixedFrequencyGetter(FrequencyGetter):
    def __init__(self, target_frequency):
        self.target_frequency = target_frequency

    def get_target_frequency(self, current_frequency: float) -> float:
        return self.target_frequency


class NearestFrequencyGetter(FrequencyGetter):
    def __init__(self, frequency_selection: Sequence[float]):
        self.frequency_selection = frequency_selection

    def get_target_frequency(self, current_frequency: float) -> float:
        return self.frequency_selection[snap_nearest_index(current_frequency, self.frequency_selection)]
