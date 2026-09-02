"""Small CLI feedback helpers."""

from itertools import cycle
import sys
from threading import Event, Thread
from time import sleep


class Spinner:
    """Show a terminal spinner while a blocking operation runs."""

    def __init__(self, message: str) -> None:
        self.message = message
        self._done = Event()
        self._thread = Thread(target=self._spin, daemon=True)

    def __enter__(self) -> "Spinner":
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._done.set()
        self._thread.join()
        status = "Failed" if exc_type else "Done"
        sys.stdout.write(f"\r{status}: {self.message.ljust(48)}\n")
        sys.stdout.flush()

    def _spin(self) -> None:
        for symbol in cycle("|/-\\"):
            if self._done.is_set():
                break
            sys.stdout.write(f"\r{symbol} {self.message}...")
            sys.stdout.flush()
            sleep(0.1)

