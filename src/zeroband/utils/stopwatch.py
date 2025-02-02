import time

from zeroband.config import Config

class Stopwatch:
    def __init__(self, config: Config | None = None):
        from zeroband.utils.logger import get_logger

        self.timers: dict[str, dict[str, float]] = {} # Timer name -> {start_time, last_lap_time}
        self.stack: list[str] = [] # List timer names in order of last constructed
        self.logger = get_logger(config)
        self.disabled = (config.log_level != "DEBUG") if config else False

    def _resolve_name(self, name: str | None) -> str:
        if name is None:
            if not self.stack:
                raise ValueError("No active timers")
            return self.stack[-1]
        return name

    def start(self, name: str) -> None:
        if self.disabled:
            return

        current_time = time.perf_counter()
        self.timers[name] = {
            'start_time': current_time,
            'last_lap_time': current_time
        }
        self.stack.append(name)

    def _lap(self, name: str | None = None) -> float:
        if self.disabled:
            return 0.0

        name = self._resolve_name(name)
        if name not in self.stack:
            raise ValueError(f"Timer '{name}' is not active")

        timer = self.timers.get(name)
        if not timer:
            raise ValueError(f"Timer '{name}' does not exist")

        current_time = time.perf_counter()
        elapsed = current_time - timer['last_lap_time']
        timer['last_lap_time'] = current_time
        return elapsed

    def start_block(self, message: str | None = None, name: str | None = None) -> None:
        if self.disabled:
            return

        self._lap(name)
        if message:
            self.logger.debug(message)

    def end_block(self, format_str: str | None = None, name: str | None = None) -> None:
        if self.disabled:
            return

        lap_time = self._lap(name)
        if not format_str:
            return
        elif "{" in format_str:
            self.logger.debug(format_str.format(name=name, time=lap_time))
        else:
            self.logger.debug(f"{format_str} in {lap_time:.2f} seconds")

    def elapsed(self, name: str | None = None) -> float:
        if self.disabled:
            return 0.0

        name = self._resolve_name(name)
        timer = self.timers.get(name)
        if not timer:
            raise ValueError(f"Timer '{name}' does not exist")

        current_time = time.perf_counter()
        return current_time - timer['start_time']

    def stop(self, name: str | None = None) -> float:
        if self.disabled:
            return 0.0

        name = self._resolve_name(name)
        elapsed = self.elapsed(name)

        if name in self.stack:
            self.stack.remove(name)
            self.timers.pop(name)

        return elapsed

    def reset(self) -> None:
        self.timers.clear()
        self.stack.clear()

