import numpy as np


class Channel:
    def __init__(self, service_rate):
        self.service_rate = service_rate
        self.service_time = float('inf')
        self.status = False     # not busy at the moment

        self.total_time = 0     # sum of service times
        self.total_served = 0   # total number of served entries

    def generate_service_time(self):
        return np.random.exponential(1. / self.service_rate)