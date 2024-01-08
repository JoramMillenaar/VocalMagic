from src.base import AudioProcessor, AudioStream


class AudioProcessingPipeline:
    def __init__(self):
        self.processors = []

    def add_processor(self, processor: AudioProcessor):
        self.processors.append(processor)

    def process(self, audio_data):
        for processor in self.processors:
            audio_data = processor.process(audio_data)
        return audio_data

    def run(self, audio_source: AudioStream):
        for audio_chunk in audio_source:
            self.process(audio_chunk)
