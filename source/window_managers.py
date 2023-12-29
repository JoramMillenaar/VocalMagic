import numpy as np

from shared.base import AudioStreamDecorator, AudioStream


class AudioOverlapper:
    def __init__(self, frequency_resolution, chunk_size, sample_rate):
        desired_chunk_length = sample_rate // frequency_resolution
        self.overlap_factor = desired_chunk_length // chunk_size
        self.overlap_size = int(chunk_size * self.overlap_factor)
        self.buffer = np.zeros(self.overlap_size, dtype=np.float32)
        self.chunk_size = self.overlap_size + chunk_size

    def concatenate(self, stream_item):
        chunk = np.concatenate((self.buffer, stream_item))
        self.buffer = chunk[-self.overlap_size:]
        return chunk


class IncreaseWindowSizeDecorator(AudioStreamDecorator):
    def __init__(self, stream: AudioStream, frequency_resolution: int):
        super().__init__(stream)
        self.overlapper = AudioOverlapper(frequency_resolution, self.chunk_size, self.sample_rate)
        self.chunk_size = self.overlapper.chunk_size

    def transform(self, stream_item):
        return self.overlapper.concatenate(stream_item)


class DecreaseWindowSizeDecorator(AudioStreamDecorator):
    def __init__(self, stream: AudioStream, desired_chunk_size: int):
        super().__init__(stream)
        self.desired_chunk_size = desired_chunk_size
        self.chunk_size = desired_chunk_size

    def transform(self, stream_item):
        return stream_item[-self.desired_chunk_size:]

