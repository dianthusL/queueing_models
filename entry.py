import numpy as np


class Entry:
    def __init__(self, arrival_rate):
        self.arrival_rate = arrival_rate
        self.arrival_time = self.generate_arrival_time()

        self.total_arrivals = 0
        self.total_wait_time = 0.0

    def generate_arrival_time(self):
        return np.random.exponential(1. / self.arrival_rate)