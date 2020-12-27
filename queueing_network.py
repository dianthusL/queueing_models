import numpy as np
from scipy import linalg
from scipy.special import factorial

import config as nc


class System:
    def __init__(self, type_, mu, channels_num):
        self.type = type_
        self.mu = mu
        self.m = channels_num


class Network:
    def __init__(self, lambdas, p_0_ir, p_r, types, service_times, channels_num):
        self.lambdas = lambdas
        self.p_0_ir = p_0_ir
        self.p_r = p_r
        self.mu_ir = 1. / service_times
        self.types = types
        self.channels_num = np.where(channels_num == np.inf, 1, channels_num)

        self.lambda_0_ir = self.lambdas * self.p_0_ir
        self.lambda_ir = self.calculate_lambda_ir()
        self.rho_ir = self.calculate_rho_ir()

        self.systems = self.generate_systems(types, self.mu_ir, self.channels_num)
        self.k_ir = self.calculate_K_ir()

    @staticmethod
    def generate_systems(types, mu_ir, channels_num):
        return [System(t, mu, int(n)) for t, mu, n in zip(types, mu_ir, channels_num)]

    def calculate_lambda_ir(self):
        lambd = []
        for i, p in enumerate(self.p_r):
            np.fill_diagonal(p, -1)
            lambd.append(linalg.solve(p.T, -self.lambda_0_ir[:, i]))
        return np.array(lambd).T

    def calculate_rho_ir(self):
        return self.lambda_ir / (self.channels_num * self.mu_ir)

    def calculate_state(self, entries_num):
        return np.prod([self.calculate_pi_k(k_i, i) for i, k_i in enumerate(entries_num)])

    def calculate_pi_k(self, k_i, i):
        s = self.systems[i]
        if s.type == 1:
            if s.m == 1:
                return (1 - self.rho_i(i)) * np.power(self.rho_i(i), k_i)
            else:
                m = s.m
                k_values = np.array(range(m))
                first = 1. / (np.sum(np.power(m * self.rho_i(i), k_values) / factorial(k_values))
                              + (np.power(m * self.rho_i(i), m) / (factorial(m) * (1 - self.rho_i(i)))))
                if k_i <= m:
                    return first * np.power(m * self.rho_i(i), k_i) / factorial(k_i)
                else:
                    return first * np.power(m, m) * np.power(self.rho_i(i), k_i) / factorial(m)
        elif s.type == 3:
            return np.exp(-self.rho_i(i)) * (np.power(self.rho_i(i), k_i) / factorial(k_i))

    def rho_i(self, i):
        return np.sum(self.rho_ir[i, :])

    def calculate_K_ir(self):
        K_ir = np.zeros(self.rho_ir.shape)
        for i, s in enumerate(self.systems):
            for j in range(self.rho_ir.shape[1]):
                if s.type == 1:
                    m_i = s.m
                    k_v = np.array(range(m_i))
                    rho_ir = self.rho_ir[i, j]
                    rho_i = self.rho_i(i)
                    K_ir[i, j] = m_i * rho_ir \
                        + (rho_ir / (1 - rho_i)) \
                        * (np.power(m_i * rho_i, m_i) / (factorial(m_i) * (1 - rho_i))) \
                        * (1 / (np.sum((np.power(m_i * rho_i, k_v)) / factorial(k_v)) + (np.power(m_i * rho_i, m_i) / factorial(m_i) * (1 / (1 - rho_i)))))
                elif s.type == 3:
                    K_ir[i, j] = self.lambda_ir[i, j] / s.mu

        return K_ir

    def calculate_T_ir(self):
        return np.divide(self.k_ir, self.lambda_ir, out=np.zeros_like(self.k_ir), where=self.lambda_ir!=0)

    def calculate_W_ir(self):
        t = 1. / np.hstack([self.mu_ir for _ in range(self.lambda_ir.shape[1])])
        t_ir = self.calculate_T_ir()
        return np.subtract(t_ir, t, out=np.zeros_like(t_ir), where=t_ir!=0)

    def calculate_Q_ir(self):
        return self.lambda_ir * self.calculate_W_ir()

    def cost_function(self, waiting_costs, unoccupied_costs):
        q_ir = self.calculate_Q_ir()[self.types!=3, :]
        m_i = np.array([s.m - self.rho_i(i) for i, s in enumerate(self.systems)])[self.types!=3]
        return np.sum(q_ir * waiting_costs) + np.sum(m_i * unoccupied_costs)

    def is_valid(self):
        return True if np.all(np.array([self.rho_i(i) for i in range(len(self.systems))]) < 1) else False


if __name__ == "__main__":
    lambdas = nc.requester_num / nc.working_time  # entry lambdas for every class

    net = Network(lambdas, nc.p_0_ir, nc.p_r, nc.system_types, nc.service_times, nc.channels_num)
    print("Throughput of each class in every system (lambda_ir):")
    print(net.lambda_ir)
    print("\nRelative service intensity of each class in every system (rho_ir):")
    print(net.rho_ir)

    print("\nNetwork states probabilities:")
    for state in nc.network_states:
        print("PI{} = {:.20f}".format(state, net.calculate_state(state)))

    print("\nAverage number of entries of each class in every system (K_ir):")
    print(net.k_ir)
