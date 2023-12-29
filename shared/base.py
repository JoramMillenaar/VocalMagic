from abc import ABC, abstractmethod
from typing import Iterator


class AudioStream(Iterator, ABC):
    def __init__(self, sample_rate: int, chunk_size: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.is_closed = False
        self.is_closing = False
        self.current = None
        self._cached_iterable = None

    @abstractmethod
    def iterable(self) -> Iterator:
        pass

    def close(self):
        self.is_closed = True

    def start_closing(self):
        self.is_closing = True
        
    def get_current(self):
        return self.current

    def __next__(self):
        if self._cached_iterable is None:
            self._cached_iterable = self.iterable()

        if self.is_closed:
            raise StopIteration

        try:
            self.current = next(self._cached_iterable)
        except (KeyboardInterrupt, StopIteration):
            self.close()
            raise
        return self.current

    def run(self):
        try:
            while True:
                next(self)
        except KeyboardInterrupt:
            self.close()
            raise

    def __del__(self):
        self.close()


class AudioStreamDecorator(AudioStream, ABC):
    def __init__(self, stream: AudioStream):
        super().__init__(sample_rate=stream.sample_rate, chunk_size=stream.chunk_size)
        self.stream = stream

    def __next__(self):
        current = super().__next__()
        return self.transform(current)

    def start_closing(self):
        self.stream.start_closing()
        super().start_closing()

    @abstractmethod
    def transform(self, stream_item):
        pass

    def iterable(self) -> Iterator:
        return self.stream

    def __copy__(self):
        new_copy = type(self)(self.stream)
        new_copy.__dict__.update({k: v for k, v in self.__dict__.items() if k != 'stream'})
        return new_copy
