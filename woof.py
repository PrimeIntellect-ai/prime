import time
import os
import multiprocessing as mp
from multiprocessing.process import _children
from zeroband.utils.shared_int_deque import SharedIntDeque

TEMPO = 5


class Foo:
    def __init__(self):
        self._beats = SharedIntDeque(5)

    def _start_heartbeat(self):
        """Start sending heartbeats to the global store in a separate process."""

        self._heartbeat_stop_event = mp.Event()
        self._heartbeat_process = mp.Process(target=self._heartbeat_loop, args=(self._heartbeat_stop_event,))
        self._heartbeat_process.start()

    def beat(self):
        self._beats.append(int(time.time()))

    def _stop_heartbeat(self):
        """Stop the heartbeat process."""
        self._send_deathrattle()
        if hasattr(self, "_heartbeat_stop_event"):
            self._heartbeat_stop_event.set()
            self._heartbeat_process.join()

    def _heartbeat_loop(self, stop_event):
        """Continuously send heartbeats until stopped."""
        try:
            while not stop_event.is_set():
                if time.time() - self._beats[-1] > TEMPO:
                    beats = self._beats.to_list()
                    beats_diff = [beats[i] - beats[i - 1] for i in range(1, len(beats))]
                    raise Exception(
                        f"Beat took {time.time() - beats[-1]}s which is longer than {TEMPO}s. Beat diffs: {beats_diff}"
                    )

                print("Heartbeat")
                time.sleep(1)
        finally:
            self._send_deathrattle()

    def _send_deathrattle(self):
        """Send a deathrattle to the global store."""
        print("Deathrattle")


def main():
    foo = Foo()
    foo._start_heartbeat()

    for i in range(10):
        foo.beat()
        time.sleep(i)


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    try:
        main()
    finally:
        for p in _children:
            p.terminate()
