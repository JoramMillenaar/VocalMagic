import threading
import wave

import numpy as np
import sounddevice as sd

from shared.base import AudioStreamDecorator, AudioStream


def array_to_wav_format(data: np.array):
    """Convert the float32 array to int16 to write to WAV file"""
    return (data * 32767).astype(np.int16).tobytes()


class AudioPlaybackDecorator(AudioStreamDecorator):
    def __init__(self, stream: AudioStream):
        super().__init__(stream)
        self.output = sd.RawOutputStream(self.sample_rate, channels=1, dtype='float32', blocksize=self.chunk_size)
        self.output.start()

        self.play_thread = threading.Thread(target=self.play, args=(np.zeros(self.chunk_size, dtype=np.float32),))
        self.play_thread.start()

    def transform(self, stream_item):
        self.play_thread.join()
        self.play_thread = threading.Thread(target=self.play, args=(stream_item,))
        self.play_thread.start()
        return stream_item

    def play(self, current):
        self.output.write(np.clip(current, -1, 1).astype(np.float32))

    def close(self):
        self.output.close()
        super().close()


class AudioFileOutputDecorator(AudioStreamDecorator):
    def __init__(self, stream: AudioStream, filename: str):
        super().__init__(stream)
        self.filename = filename
        self.wav_file = wave.open(self.filename, 'wb')
        self.wav_file.setnchannels(1)
        self.wav_file.setsampwidth(2)  # 2 bytes for 'int16' format
        self.wav_file.setframerate(self.sample_rate)

        self.write_thread = threading.Thread(target=self.write_to_file, args=(b'',))
        self.write_thread.start()

    def transform(self, stream_item):
        data = array_to_wav_format(stream_item)
        self.write_thread.join()
        self.write_thread = threading.Thread(target=self.write_to_file, args=(data,))
        self.write_thread.start()
        return stream_item

    def write_to_file(self, data):
        self.wav_file.writeframes(data)

    def close(self):
        self.write_thread.join()  # Ensure the last bit of audio is written
        self.wav_file.close()
        super().close()
