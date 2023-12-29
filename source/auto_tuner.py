from shared.base import AudioStreamDecorator, AudioStream
from source.frequency_extractor import FrequencyExtractor
from source.analytics import RangedAudioAnalyser
from source.manipulators import Manipulator
from source.window_managers import AudioOverlapper


class AudioModifierDecorator(AudioStreamDecorator):
    def __init__(self,
                 stream: AudioStream,
                 extractor: FrequencyExtractor,
                 frequency_resolution: int,
                 manipulator: Manipulator,
                 ):
        super().__init__(stream)
        self.extractor = extractor
        self.frequency_resolution = frequency_resolution
        self.overlapper = AudioOverlapper(self.frequency_resolution, self.chunk_size, self.sample_rate)
        self.analyser = RangedAudioAnalyser(self.overlapper.chunk_size, self.sample_rate)
        self.manipulator = manipulator

    def transform(self, stream_item):
        overlapped_audio = self.overlapper.concatenate(stream_item)
        analytics = self.analyser.analyse(overlapped_audio)
        wave_id = self.extractor.extract_wave_id(analytics)
        stream_item = self.manipulator.manipulate(overlapped_audio, analytics, wave_id)
        return stream_item[-self.chunk_size:]
