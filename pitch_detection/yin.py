import numpy as np


def difference_function(audio_chunk, lag):
    if lag == 0:
        return 0.0
    diff = audio_chunk[:-lag] - audio_chunk[lag:]
    return np.sum(diff[:len(audio_chunk) // 2 - lag] ** 2)


def get_running_average(diff, window_size):
    indices = np.arange(1, window_size)
    cumulative_sums = np.cumsum(diff[1:window_size])
    normalization_factors = 1 / indices
    return np.concatenate(([1], diff[1:window_size] / (cumulative_sums * normalization_factors)))


def get_wave_duration(running_average, threshold=0.1):
    for i in range(1, len(running_average)):
        if running_average[i] < threshold:
            while i + 1 < len(running_average) and running_average[i + 1] < running_average[i]:
                i += 1
            return i
    return -1  # No fundamental frequency found


def quadratic_interpolation(prev_point, mid_point, next_point):
    return 0.5 * (prev_point - next_point) / (prev_point - 2 * mid_point + next_point)


def yin_pitch_detection(audio_chunk: np.ndarray, sample_rate: int, threshold: float = 0.1):
    window = len(audio_chunk) // 2
    diff = [difference_function(audio_chunk, lag) for lag in range(window)]
    running_avg = get_running_average(diff, window)
    wave_duration = get_wave_duration(running_avg, threshold)

    if wave_duration == -1:
        return None  # No pitch found

    if 1 < wave_duration < window - 1:
        wave_duration += quadratic_interpolation(*running_avg[wave_duration - 1: wave_duration + 2])

    return sample_rate / wave_duration
