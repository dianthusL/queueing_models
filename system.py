import numpy as np
import pandas as pd
from scipy.special import factorial

from entry import Entry
from channel import Channel


class System:
    def __init__(self, arrival_rate, num_channels, service_rates):
        self.arrival_rate = arrival_rate
        self.service_rates = service_rates
        self.num_channels = num_channels
        self.entry = Entry(arrival_rate)
        self.channels = self.generate_channels(service_rates)

        self.num_in_queue = 0   # actual number of entries in the queue
        self.total_in_queue = 0 # total entries in the queue

    @staticmethod
    def generate_channels(service_rates):
        return [Channel(rate) for rate in service_rates]

    def available_channels(self):
        return [c for c in self.channels if not c.status]

    def calculate_parameters(self):
        lambd = self.arrival_rate
        m = self.num_channels

        # handle M/M/c as multi server system with constant service rate for all channels
        if np.all(self.service_rates == self.service_rates[0]):
            mu = self.service_rates[0]
            k_values = np.arange(0, m)

            # relative service intensity
            rho = lambd / mu
            assert lambd < m * mu
            # probability that there is zero entries in the system
            p_0 = 1. / (np.sum(np.power(rho, k_values) / factorial(k_values)) + (np.power(rho, m) / (factorial(m-1) * (m - rho))))
            # average service time
            t = 1. / mu
            # average number of entries in the system
            K = rho + (np.power(rho, m+1) / (np.power(m-rho, 2) * factorial(m-1))) * p_0
            # average time of entry's residing in the system
            T = K / lambd
            # average number of entries in the queue
            Q = np.power(rho, m+1) / (np.power(m-rho, 2) * factorial(m-1)) * p_0
            # average time of entry's residing in the queue
            W = Q / lambd

            values = [lambd, mu, rho, t, K, T, Q, W]
            params = ['lambda', 'mu', 'rho', 't', 'K', 'T', 'Q', 'W']

        # handle different service rates
        else:
            mu_k = self.service_rates
            k_values = np.arange(0, m)
            rho_k = lambd / mu_k

            # assert lambd < mu_k
            t_k = 1. / mu_k
            K = np.nan
            T = np.nan
            Q = np.nan
            W = np.nan

            values = [lambd, *mu_k, *rho_k, *t_k, K, T, Q, W]
            params = ['lambda',
                      *[f'mu_{i}' for i in range(1, m+1)],
                      *[f'rho_{i}' for i in range(1, m+1)],
                      *[f't_{i}' for i in range(1, m+1)],
                      'K', 'T', 'Q', 'W']

        df = pd.DataFrame({'parameters': params,
                           'calculated': values
                           })
        df.set_index(['parameters'], inplace=True)

        # print(df)
        return df
