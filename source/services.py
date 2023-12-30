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


def resample_audio(stream_item, chunk_size, factor):
    return resample(stream_item, num=int(chunk_size * factor))


def base_frequency_indexes(frequencies: np.ndarray, lenience: float = 0.0):
    base_indices = [None] * len(frequencies)

    for i, freq in enumerate(frequencies):
        for j in range(i):
            base_freq = frequencies[j]
            if base_freq <= freq / 2:
                if abs(freq - (base_freq * round(freq / base_freq))) <= lenience:
                    base_indices[i] = j
                    break

    return base_indices


def loudest_base_frequency_index(base_indices, magnitudes):
    harmonic_sums = {}

    for i, base_index in enumerate(base_indices):
        if base_index is not None:
            harmonic_sums.setdefault(base_index, 0)
            harmonic_sums[base_index] += magnitudes[i]
        else:
            harmonic_sums[base_index] = magnitudes[i]

    return max(harmonic_sums, key=harmonic_sums.get)


def loudest_harmonic_of_loudest_base(base_indices, magnitudes):
    harmonic_sums = {}
    loudest_harmonics = {}

    for i, base_index in enumerate(base_indices):
        if base_index is not None:
            harmonic_sums.setdefault(base_index, 0)
            harmonic_sums[base_index] += magnitudes[i]

            if base_index not in loudest_harmonics or magnitudes[i] > magnitudes[loudest_harmonics[base_index]]:
                loudest_harmonics[base_index] = i

    if not harmonic_sums:
        return None

    loudest_base_index = max(harmonic_sums, key=harmonic_sums.get)
    return loudest_harmonics[loudest_base_index]


def snap_nearest_index(value: float, options: Sequence[float]) -> int:
    nearest_index = 0
    smallest_diff = abs(value - options[0])

    for i, option in enumerate(options[1:], start=1):
        current_diff = abs(value - option)

        if current_diff < smallest_diff:
            nearest_index = i
            smallest_diff = current_diff

    return nearest_index


def adjust_chunk_size(chunk, chunk_size):
    if len(chunk) < chunk_size:
        return np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
    if len(chunk) > chunk_size:
        return chunk[:chunk_size]
    return chunk


NOTE_FREQUENCIES = tuple(calculate_frequency(key) for key in range(16, 89))
