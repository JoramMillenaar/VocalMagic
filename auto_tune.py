import argparse

from AudioIO.input_streams import WAVFileReadStream, MicrophoneStream
from AudioIO.output_streams import AudioPlaybackProcessor, WAVFileWriteStream
from pypitch.detectors import YinPitchDetector

from src.frequency_getters import NearestFrequencyGetter
from src.pipelines import AudioProcessingPipeline
from src.pitch_shifter import PitchShifter
from src.services import NOTE_FREQUENCIES


def parse_args():
    parser = argparse.ArgumentParser(description='Auto-tune application with microphone input and audio playback.')

    parser.add_argument('--sample-rate', type=int, default=44100, help='Sample rate for audio processing')
    parser.add_argument('--chunk-size', type=int, default=512, help='Chunk size for audio processing')
    parser.add_argument(
        '--threshold', type=float, default=0.1,
        help='Pitch detection sensitivity. Lower values increase sensitivity but conversely lead to false positives'
    )
    parser.add_argument(
        '--source-file', type=str, default='', help='Source WAV file to process. If empty, uses microphone input.'
    )
    parser.add_argument(
        '--output-file', type=str, default='', help='Output WAV file to write. If empty, plays back audio.'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Select audio source based on the provided argument
    if args.source_file:
        audio_source = WAVFileReadStream(args.source_file, chunk_size=args.chunk_size)
        args.sample_rate = audio_source.sample_rate
    else:
        audio_source = MicrophoneStream(sample_rate=args.sample_rate, chunk_size=args.chunk_size)

    pitch_shifter = PitchShifter(
        frequency_getter=NearestFrequencyGetter(NOTE_FREQUENCIES),
        pitch_detector=YinPitchDetector(args.sample_rate)
    )

    pipeline = AudioProcessingPipeline()
    pipeline.add_processor(pitch_shifter)

    # Select output destination based on the provided argument
    if args.output_file:
        audio_output = WAVFileWriteStream(args.output_file, args.sample_rate, audio_source.channels)
        pipeline.add_processor(audio_output)
    else:
        speaker_output = AudioPlaybackProcessor(
            sample_rate=args.sample_rate, chunk_size=args.chunk_size, channels=audio_source.channels
        )
        pipeline.add_processor(speaker_output)

    pipeline.stream(audio_source=audio_source)


if __name__ == '__main__':
    main()
