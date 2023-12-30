from source.base import AudioProcessor


class AudioProcessingPipeline:
    def __init__(self):
        self.processors = []

    def add_processor(self, processor: AudioProcessor):
        self.processors.append(processor)

    def process(self, audio_data):
        for processor in self.processors:
            audio_data = processor.process(audio_data)
        return audio_data
