from source.base import AudioProcessor
from source.pitch_detectors import PitchDetector
from source.manipulators import Manipulator
from source.window_managers import AudioOverlapProcessor


class AudioModifierProcessor(AudioProcessor):
    def __init__(self,
                 pitch_detector: PitchDetector,
                 frequency_resolution: int,
                 manipulator: Manipulator,
                 chunk_size,
                 sample_rate
                 ):
        super().__init__(sample_rate)
        self.pitch_detector = pitch_detector
        self.frequency_resolution = frequency_resolution
        self.overlapper = AudioOverlapProcessor(self.frequency_resolution, chunk_size, sample_rate)
        self.manipulator = manipulator
        self.chunk_size = chunk_size

    def process(self, stream_item):
        overlapped_audio = self.overlapper.process(stream_item)
        wave_id = self.pitch_detector.extract_wave_id(overlapped_audio)
        stream_item = self.manipulator.manipulate(overlapped_audio, wave_id)
        return stream_item[-self.chunk_size:]
