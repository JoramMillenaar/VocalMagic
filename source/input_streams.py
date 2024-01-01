import wave
from typing import Iterator

import numpy as np
import sounddevice as sd

from source.base import AudioStream


class MicrophoneStream(AudioStream):
    def __init__(self, chunk_size: int, sample_rate: int, channels: int = 1):
        super().__init__(sample_rate=sample_rate)
        self.chunk_size = chunk_size
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


class WAVFileReadStream(AudioStream):
    def __init__(self, file_path: str, chunk_size: int):
        self.file_path = file_path
        self.wav_file = wave.open(self.file_path, 'rb')
        self.channels = self.wav_file.getnchannels()
        self.chunk_size = chunk_size
        super().__init__(sample_rate=self.wav_file.getframerate())

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
