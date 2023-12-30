import argparse

from shared.outputs import AudioPlaybackDecorator
from source.auto_tuner import AudioModifierDecorator
from source.pitch_detectors import SimplePitchDetector
from source.input_streams import MicrophoneStream
from source.manipulators import AutoTuneManipulator
from source.services import NOTE_FREQUENCIES


def parse_args():
    parser = argparse.ArgumentParser(description='Auto-tune application with microphone input and audio playback.')

    parser.add_argument('--sample-rate', type=int, default=44100, help='Sample rate for audio processing')
    parser.add_argument('--chunk-size', type=int, default=512, help='Chunk size for audio processing')
    parser.add_argument(
        '--frequency-resolution', type=int, default=4,
        help='Frequency resolution for auto tuner. Default is 4, meaning that the precision is in steps of four'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    s = MicrophoneStream(sample_rate=args.sample_rate, chunk_size=args.chunk_size)
    s = AudioModifierDecorator(
        stream=s,
        pitch_detector=SimplePitchDetector(sample_rate=args.sample_rate),
        frequency_resolution=args.frequency_resolution,
        manipulator=AutoTuneManipulator(args.sample_rate, NOTE_FREQUENCIES)
    )
    s = AudioPlaybackDecorator(s)
    s.run()


if __name__ == '__main__':
    main()
