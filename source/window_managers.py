import numpy as np
from scipy.signal import butter, lfilter

from source.base import AudioProcessor


class AudioOverlapProcessor(AudioProcessor):
    def __init__(self, frequency_resolution, sample_rate):
        super().__init__(sample_rate)
        self.desired_chunk_length = sample_rate // frequency_resolution

        self.overlap_factor = None
        self.overlap_size = None
        self.buffer = None
        self.chunk_size = None

    def prime(self, chunk_size):
        self.overlap_factor = self.desired_chunk_length // chunk_size
        self.overlap_size = int(chunk_size * self.overlap_factor)
        self.buffer = np.zeros(self.overlap_size, dtype=np.float32)
        self.chunk_size = self.overlap_size + chunk_size

    def process(self, stream_item):
        if self.buffer is None:
            self.prime(len(stream_item))
        chunk = np.concatenate((self.buffer, stream_item))
        self.buffer = chunk[-self.overlap_size:]
        return chunk


class DecreaseWindowSizeProcessor(AudioProcessor):
    def __init__(self, desired_chunk_size: int, sample_rate):
        super().__init__(sample_rate)
        self.desired_chunk_size = desired_chunk_size
        self.chunk_size = desired_chunk_size

    def process(self, stream_item):
        return stream_item[-self.desired_chunk_size:]


class LowPassFilter(AudioProcessor):
    def __init__(self, cutoff_frequency, sample_rate, order=5):
        super().__init__(sample_rate)
        self.cutoff_frequency = cutoff_frequency
        self.order = order
        self.b, self.a = butter(self.order, self.cutoff_frequency / (0.5 * sample_rate), btype='low')

    def process(self, audio_chunk):
        return lfilter(self.b, self.a, audio_chunk)
