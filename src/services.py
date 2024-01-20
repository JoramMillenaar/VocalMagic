from typing import Sequence

import numpy as np
from scipy.signal import resample


def calculate_frequency(key: int, reference_key=49, reference_freq=440.0) -> float:
    """
    Calculate the frequency of a musical note given its key number on the piano.

    Args:
    key (int): The key number (e.g., A4 is 49, C4 is 40).
    reference_key (int): The reference key number (A4 is 49).
    reference_freq (float): The frequency of the reference key (A4 is 440 Hz).

    Returns:
    float: The frequency of the note.
    """
    return reference_freq * 2 ** ((key - reference_key) / 12)


def snap_nearest_index(value: float, options: Sequence[float]) -> int:
    nearest_index = 0
    smallest_diff = abs(value - options[0])

    for i, option in enumerate(options[1:], start=1):
        current_diff = abs(value - option)

        if current_diff < smallest_diff:
            nearest_index = i
            smallest_diff = current_diff

    return nearest_index


def resample_to_size(audio_chunk: np.array, factor: float):
    return resample(audio_chunk, num=int(len(audio_chunk) * factor))


def generate_sine_wave(freq: float, chunk_size: int, sample_rate: int, volume: float):
    t = 0
    omega = 2 * np.pi * freq
    while True:
        samples = np.arange(t, t + chunk_size, dtype=np.float32) / sample_rate
        chunk = np.sin(omega * samples) * volume
        yield chunk
        t += chunk_size


NOTE_FREQUENCIES = tuple(calculate_frequency(key) for key in range(16, 89))
