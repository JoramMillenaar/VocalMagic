from shared.base import AudioStreamDecorator, AudioStream
from source.pitch_detectors import PitchDetector
from source.manipulators import Manipulator
from source.window_managers import AudioOverlapper


class AudioModifierDecorator(AudioStreamDecorator):
    def __init__(self,
                 stream: AudioStream,
                 pitch_detector: PitchDetector,
                 frequency_resolution: int,
                 manipulator: Manipulator,
                 ):
        super().__init__(stream)
        self.pitch_detector = pitch_detector
        self.frequency_resolution = frequency_resolution
        self.overlapper = AudioOverlapper(self.frequency_resolution, self.chunk_size, self.sample_rate)
        self.manipulator = manipulator

    def transform(self, stream_item):
        overlapped_audio = self.overlapper.concatenate(stream_item)
        wave_id = self.pitch_detector.extract_wave_id(overlapped_audio)
        stream_item = self.manipulator.manipulate(overlapped_audio, wave_id)
        return stream_item[-self.chunk_size:]
