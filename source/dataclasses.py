from dataclasses import dataclass

import numpy as np


@dataclass
class WaveID:
    frequency: float
    amplitude: float
    phase: float = 0


@dataclass
class AudioAnalytics:
    chunk_size: int
    sample_rate: int

    frequency_range: np.ndarray
    complex_spectrum: np.ndarray
    magnitudes: np.ndarray
