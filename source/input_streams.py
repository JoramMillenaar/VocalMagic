import wave
from typing import Iterator

import numpy as np
import sounddevice as sd

from shared.base import AudioStream


class MicrophoneStream(AudioStream):
    def __init__(self, chunk_size: int, sample_rate: int, channels: int = 1):
        super().__init__(sample_rate=sample_rate, chunk_size=chunk_size)
        self.channels = channels
        self.stream = None

    def open_stream(self):
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            channels=self.channels
        )
        self.stream.start()

    def iterable(self) -> Iterator:
        if self.stream is None:
            self.open_stream()

        while not self.is_closed:
            r = np.squeeze(self.stream.read(self.chunk_size)[0])
            yield r

    def close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        super().close()


class ComplexSineWaveGenerator:
    def __init__(self, freq: float, sample_rate: int, chunk_size: int):
        self.freq = freq
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.t = 0  # Current time index
        self.phase_offset = 0  # Phase offset in radians

        self.current = None

    @property
    def omega(self):
        return 2 * np.pi * self.freq

    def __next__(self):
        samples = np.arange(self.t, self.t + self.chunk_size, dtype=np.float32) / self.sample_rate
        self.t += self.chunk_size
        return np.sin(self.omega * samples)

    def __iter__(self):
        return self


class ComplexSineWaveStream(AudioStream):
    def __init__(self, chunk_size: int, sample_rate: int):
        super().__init__(sample_rate=sample_rate, chunk_size=chunk_size)
        self.amplitude = 0
        self._frequency = 0
        self._gen_cache = ComplexSineWaveGenerator(0, self.sample_rate, self.chunk_size)

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        self._frequency = value
        self._gen_cache.freq = value

    def iterable(self):
        for audio_chunk in self._gen_cache:
            yield audio_chunk * self.amplitude


class WAVFileReadStream(AudioStream):
    def __init__(self, file_path: str, chunk_size: int):
        self.file_path = file_path
        self.wav_file = wave.open(self.file_path, 'rb')
        self.channels = self.wav_file.getnchannels()
        super().__init__(sample_rate=self.wav_file.getframerate(), chunk_size=chunk_size)

    def iterable(self) -> Iterator:
        while True:
            frames = self.wav_file.readframes(self.chunk_size)
            if not frames:
                break

            # Convert byte data to numpy array
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            data /= 32768  # Convert from int16 to float32 range [-1, 1]

            # Handle mono/stereo
            if self.channels > 1:
                data = np.reshape(data, (-1, self.channels))

            yield data

    def close(self):
        self.wav_file.close()
        super().close()


class ReadStream(AudioStream):
    def __init__(self, read_stream: AudioStream):
        super().__init__(sample_rate=read_stream.sample_rate, chunk_size=read_stream.chunk_size)
        self.read_stream = read_stream

    def iterable(self) -> Iterator:
        while True:
            yield self.read_stream.current


class RandomNoiseStream(AudioStream):
    def __init__(self, chunk_size: int, sample_rate: int, amplitude=1.0):
        super().__init__(sample_rate=sample_rate, chunk_size=chunk_size)
        self.amplitude = amplitude

    def iterable(self):
        while True:
            # Generate random noise
            noise = np.random.randn(self.chunk_size) * self.amplitude
            yield noise


