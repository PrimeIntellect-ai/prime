import time

from zeroband.config import Config

class Stopwatch:
    def __init__(self, config: Config | None = None):
        from zeroband.utils.logger import get_logger

        self.timers: dict[str, dict[str, float]] = {} # Timer name -> {start_time, last_lap_time}
        self.stack: list[str] = [] # List timer names in order of last constructed
        self.logger = get_logger(config)

    def _resolve_name(self, name: str | None) -> str:
        if name is None:
            if not self.stack:
                raise ValueError("No active timers")
            return self.stack[-1]
        return name

    def start(self, name: str) -> None:
        current_time = time.perf_counter()
        self.timers[name] = {
            'start_time': current_time,
            'last_lap_time': current_time
        }
        self.stack.append(name)

    def lap(self, name: str | None = None) -> float:
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
    
    def log_lap(self, name: str | None = None, format_str: str | None = None) -> float:
        lap_time = self.lap(name)
        if format_str:
            self.logger.debug(format_str.format(name=name, lap_time=lap_time))
        else:
            self.logger.debug(f"{name} Lap time: {lap_time:.2f} seconds")
        return lap_time

    def stop(self, name: str | None = None) -> float:
        name = self._resolve_name(name)
        if name not in self.stack:
            raise ValueError(f"Timer '{name}' is not active")
        self.stack.remove(name)

        timer = self.timers.get(name)
        if not timer:
            raise ValueError(f"Timer '{name}' does not exist")

        current_time = time.perf_counter()
        elapsed = current_time - timer['start_time']
        del self.timers[name]
        return elapsed

    def reset(self) -> None:
        self.timers.clear()
        self.stack.clear()





if __name__ == "__main__":
    sw = Stopwatch()
    begin_time = sw.start("Outer loop")

    for j in range(5):
        time.sleep(.1)
        lap_time = sw.lap() # Should be about .1, uses outer loop timer
        print(f"Loop {j} took {lap_time:.2f} seconds")

    for j in range(5):
        sw.start("Inner loop")
        for k in range(5):
            time.sleep(.1)
            sw.lap() # uses inner loop timer
            print(f"Inner loop {k} took {lap_time:.2f} seconds")
        time_for_inner_loops = sw.stop() # Should be about .5
        print(f"Inner loops took {time_for_inner_loops:.2f} seconds")

    time_elapsed = sw.stop("Outer loop")
    print(f"Outer loop took {time_elapsed:.2f} seconds")
