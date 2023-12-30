from dataclasses import dataclass


@dataclass
class WaveID:
    frequency: float
    amplitude: float
    phase: float = 0
