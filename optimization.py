import os
import numpy as np
import math
from random import randint, choice
import matplotlib.pyplot as plt

import config as nc
from queueing_network import Network


class Cell:
    def __init__(self, channels, net, value):
        self.channels = channels
        self.net = net
        self.value = value

    def __repr__(self):
        return f"{self.channels.T}, value: {self.value}"


class ClonAlg:
    def __init__(self, min_num, max_num, population_size, selection_size, clone_rate, mutation_rate, iterations):
        self.min_num = min_num
        self.max_num = max_num
        self.population_size = population_size
        self.selection_size = selection_size
        self.clone_rate = clone_rate
        self.mutation_rate = mutation_rate
        self.iterations = iterations

    def create_cell(self, considered_channels_num):
        channels = np.array([randint(self.min_num, self.max_num) for _ in range(considered_channels_num)])
        channels = np.concatenate((np.array([np.inf, np.inf]), channels)).reshape(-1, 1)
        net = Network(nc.requester_num / nc.working_time, nc.p_0_ir, nc.p_r, nc.system_types, nc.service_times,
                      channels)
        if net.is_valid():
            return Cell(channels, net, net.cost_function(nc.C_ir, nc.C_i))
        else:
            return self.create_cell(considered_channels_num)

    def generate_population(self, c_num):
        return [self.create_cell(c_num) for _ in range(self.population_size)]

    def select(self, population):
        return sorted(population, key=lambda cell: cell.value)[:self.selection_size]

    def clone(self, selected):
        clones = []
        max_value = max(selected, key=lambda cell: cell.value).value
        for cell in selected:
            num = math.ceil((max_value - cell.value) / max_value * self.clone_rate) + 1
            clones += [cell for _ in range(num)]
        return clones

    def hypermutate(self, clones):
        mutated = []
        max_value = max(clones, key=lambda cell: cell.value).value
        min_value = min(clones, key=lambda cell: cell.value).value
        for clone in clones:
            offset = math.ceil((clone.value - min_value) / max_value * self.mutation_rate) + 1
            mutated.append(self.mutate(clone.channels[clone.channels != np.inf], offset))
        return mutated

    def mutate(self, channels, offset):
        new_channels = []
        for c_num in channels:
            value = randint(self.min_num, offset)
            if c_num + value > self.max_num:
                new_channels.append(self.max_num)
            elif c_num - value < self.min_num:
                new_channels.append(self.min_num)
            else:
                way = choice([0, 1])
                if way:
                    new_channels.append(c_num + value)
                else:
                    new_channels.append(c_num - value)

        new_channels = np.concatenate((np.array([np.inf, np.inf]), np.array(new_channels))).reshape(-1, 1)
        net = Network(nc.requester_num / nc.working_time, nc.p_0_ir, nc.p_r, nc.system_types, nc.service_times,
                      new_channels)
        if net.is_valid():
            return Cell(new_channels, net, net.cost_function(nc.C_ir, nc.C_i))
        else:
            return self.mutate(channels, offset)

    def replace(self, population, mutated):
        population += mutated
        return sorted(population, key=lambda cell: cell.value)[:self.population_size]

    @staticmethod
    def show(population):
        for cell in population:
            print(cell)

    def summarize(self, worst, avg, best, save=True):
        fig, ax = plt.subplots(1, 1)
        ax.scatter(range(self.iterations), worst, label='Worst')
        ax.scatter(range(self.iterations), avg, label='Average')
        ax.scatter(range(self.iterations), best, label='Best')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost value")
        ax.legend()
        plt.show()
        if save:
            filename = f"min={self.min_num}_max={self.max_num}_p_size={self.population_size}_s_size={self.selection_size}" \
                       f"_c_rate={self.clone_rate}_m_rate={self.mutation_rate}_iter={self.iterations}.png"
            fig.savefig(os.path.join(os.getcwd(), 'results', filename), dpi=fig.dpi)

    def run(self, c_num=7):
        population = self.generate_population(c_num)
        worst, avg, best = [], [], []
        for _ in range(self.iterations):
            temp = sorted(population, key=lambda cell: cell.value)
            # self.show(population)
            worst.append(temp[-1].value)
            best.append(temp[0].value)
            avg.append(sum([c.value for c in population]) / self.population_size)
            selected = self.select(population)
            clones = self.clone(selected)
            mutated = self.hypermutate(clones)
            population = self.replace(population, mutated)

        sol = sorted(population, key=lambda cell: cell.value)[0]
        print(sol)
        self.summarize(worst, avg, best, save=False)


if __name__ == "__main__":
    alg = ClonAlg(nc.min_num, nc.max_num, nc.population_size, nc.selection_size, nc.clone_rate, nc.mutation_rate,
                  nc.iterations)
    alg.run()
