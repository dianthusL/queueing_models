import os
import random
import pandas as pd
import numpy as np

from system import System


class Simulation:
    def __init__(self, end_time, arrival_rate, num_channels, service_rates):
        self.clock = 0.0
        self.end_time = end_time
        self.system = System(arrival_rate, num_channels, service_rates)
        self.num = num_channels

    def advance_time(self):
        next_event_time = self.next_event()
        self.system.entry.total_wait_time += self.system.num_in_queue * (next_event_time - self.clock)
        self.clock = next_event_time

        if next_event_time == self.system.entry.arrival_time:
            self.handle_arrival_events()
        else:
            self.handle_service_events()

    def next_event(self):
        event_times = [c.service_time for c in self.system.channels]
        event_times.append(self.system.entry.arrival_time)
        return min(event_times)

    def handle_arrival_events(self):
        self.system.entry.total_arrivals += 1

        if self.system.num_in_queue == 0:
            channels = self.system.available_channels()
            if channels:
                channel = random.choice(channels)
                service_time = channel.generate_service_time()
                channel.status = True
                channel.total_time += service_time
                channel.service_time = self.clock + service_time
            else:
                self.system.num_in_queue += 1
                self.system.total_in_queue += 1
        else:
            self.system.num_in_queue += 1
            self.system.total_in_queue += 1

        self.system.entry.arrival_time = self.clock + self.system.entry.generate_arrival_time()

    def handle_service_events(self):
        channel = min(self.system.channels, key=lambda c: c.service_time)
        channel.total_served += 1

        if self.system.num_in_queue:
            service_time = channel.generate_service_time()
            channel.total_time += service_time
            channel.service_time = self.clock + service_time
            self.system.num_in_queue -= 1
        else:
            channel.service_time = float('inf')
            channel.status = False

    def run(self):
        while self.clock <= self.end_time:
            self.advance_time()

    def summarize(self, to_csv=False):
        avg_lambda = 1. / (self.clock / self.system.entry.total_arrivals)
        avg_mu_k = np.array([1. / (c.total_time / c.total_served) for c in self.system.channels])
        avg_mu = np.sum(avg_mu_k) / self.system.num_channels
        avg_rho_k = avg_lambda / avg_mu_k
        avg_rho = np.sum(avg_rho_k) / self.system.num_channels
        avg_t_k = [c.total_time / c.total_served for c in self.system.channels]
        avg_t = np.sum(avg_t_k) / self.system.num_channels
        avg_K = avg_lambda * (self.system.entry.total_wait_time / sum([c.total_served for c in self.system.channels]) +
                              sum([c.total_time / c.total_served for c in self.system.channels]) / self.system.num_channels)
        avg_T = self.system.entry.total_wait_time / sum([c.total_served for c in self.system.channels]) + \
                sum([c.total_time / c.total_served for c in self.system.channels]) / self.system.num_channels
        avg_Q = avg_lambda * (self.system.entry.total_wait_time / sum([c.total_served for c in self.system.channels]))
        avg_W = self.system.entry.total_wait_time / sum([c.total_served for c in self.system.channels])

        df = self.system.calculate_parameters()

        if np.all(self.system.service_rates == self.system.service_rates[0]):
            df['simulated'] = [avg_lambda, avg_mu, avg_rho, avg_t, avg_K, avg_T, avg_Q, avg_W]
        else:
            df['simulated'] = [avg_lambda, *avg_mu_k, *avg_rho_k, *avg_t_k, avg_K, avg_T, avg_Q, avg_W]

        print(df)
        if to_csv:
            df.to_csv(os.path.join(os.getcwd(), 'results', 'out.csv'))

lambd = 15
channels_num = 6
total_time = 8*60
mu_s_const = np.array([4, 4, 4, 4, 4, 4])
mu_s_diff = np.array([4, 5, 6, 7, 8, 3])

s = Simulation(total_time, lambd, channels_num, mu_s_diff)
s.run()
s.summarize(True)
