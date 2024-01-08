import numpy as np
from scipy.signal import butter, lfilter

from src.base import AudioProcessor


class AudioOverlapProcessor(AudioProcessor):
    def __init__(self, output_chunk_size, sample_rate):
        self.sample_rate = sample_rate
        self.output_chunk_size = output_chunk_size

        self.overlap_factor = None
        self.overlap_size = None
        self.buffer = None
        self.chunk_size = None

    def prime(self, chunk_size):
        self.overlap_factor = self.output_chunk_size // chunk_size
        self.overlap_size = int(chunk_size * self.overlap_factor)
        self.buffer = np.zeros(self.overlap_size, dtype=np.float32)
        self.chunk_size = self.overlap_size + chunk_size

    def process(self, audio_chunk):
        if self.buffer is None:
            self.prime(len(audio_chunk))
        chunk = np.concatenate((self.buffer, audio_chunk))
        self.buffer = chunk[-self.overlap_size:]
        return chunk


class DecreaseWindowSizeProcessor(AudioProcessor):
    def __init__(self, desired_chunk_size: int):
        self.desired_chunk_size = desired_chunk_size
        self.chunk_size = desired_chunk_size

    def process(self, audio_chunk):
        return audio_chunk[-self.desired_chunk_size:]


class LowPassFilter(AudioProcessor):
    def __init__(self, cutoff_frequency, sample_rate, order=5):
        self.sample_rate = sample_rate
        self.cutoff_frequency = cutoff_frequency
        self.order = order
        self.b, self.a = butter(self.order, self.cutoff_frequency / (0.5 * sample_rate), btype='low')

    def process(self, audio_chunk):
        return lfilter(self.b, self.a, audio_chunk)


class MonoAudioProcessor(AudioProcessor):
    def process(self, audio_chunk):
        if len(audio_chunk.shape) > 1 and audio_chunk.shape[1] > 1:
            mono_data = np.mean(audio_chunk, axis=1)
        else:
            mono_data = audio_chunk
        return mono_data


class SimpleNoiseGateAudioProcessor(AudioProcessor):
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        if np.sum(np.abs(audio_chunk)) / len(audio_chunk) > self.threshold:
            return audio_chunk
        return np.zeros(len(audio_chunk), dtype=np.float32)


class BandPassFilterAudioProcessor(AudioProcessor):
    def __init__(self, sample_rate, low_cut=80, high_cut=1100, order=5):
        self.sample_rate = sample_rate
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.order = order
        self.b, self.a = butter(self.order, [self.low_cut, self.high_cut], btype='band', fs=sample_rate)

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        return lfilter(self.b, self.a, audio_chunk)
