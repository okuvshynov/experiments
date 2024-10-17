import time
from typing import List, Tuple

class RateLimiter:
    def __init__(self, tokens_rate: int, period: int):
        self.tokens_rate: int = tokens_rate
        self.period: int = period
        self.history: List[Tuple[float, int]] = []

    def wait_time(self) -> float:
        total_size = sum(size for _, size in self.history)
        if total_size < self.tokens_rate:
            return 0
        current_time = time.time()
        running_total = total_size
        for time_stamp, size in self.history:
            running_total -= size
            if running_total <= self.tokens_rate:
                return max(0, time_stamp + self.period - current_time)
        return 0

    def add_request(self, size: int):
        current_time = time.time()
        self.history = [(t, s) for t, s in self.history if current_time - t <= self.period]
        self.history.append((current_time, size))

    def wait(self):
        wait_for = self.wait_time()
        if wait_for > 0:
            time.sleep(wait_for)