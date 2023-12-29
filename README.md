<p align="center">
    <img src="logo.png" alt="drawing" width="250" />
</p>

# VocalMagic

VocalMagic is an innovative audio processing tool designed to transform and enhance vocal performances in real-time. Inspired by the mesmerizing sound of Bon Iver's Messina, this project aspires to bring the magic of professional vocal modification to your fingertips. 

## Features

- **Auto-Tuner:** The initial release includes an auto-tuner that adjusts your vocal pitch to the nearest note, ensuring your singing is harmoniously aligned with standard musical scales.

## Getting Started

### Prerequisites

- Python 3.x
- NumPy, SciPy, and SoundDevice packages

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/VocalMagic.git
   ```

2. Navigate to the cloned directory:

   ```sh
   cd VocalMagic
   ```

3. Install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

### Usage

Run the `auto_tune.py` script with optional command-line arguments to start the auto-tuner:

```sh
python auto_tune.py --sample-rate 44100 --chunk-size 1024 --frequency-resolution 4
```

#### Command-Line Arguments

- `--sample-rate`: Sample rate for audio processing (default: 44100)
- `--chunk-size`: Chunk size for audio processing (default: 512)
- `--frequency-resolution`: Frequency resolution for the auto tuner (default: 4)

## Contributions

VocalMagic is an open-source project, and contributions are warmly welcomed. Whether you have ideas for new features, improvements, or bug fixes, feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- Inspired by Bon Iver's Messina
