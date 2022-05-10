import pandas as pd
from numba import njit
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import os

def non_jit_performance(list_ndarray):
    def sum(x):
        return x + 1
    def square(x):
        return x ** 2
    func = (sum, square)
    time = []
    for i in list_ndarray:
        start = timer()
        s = pd.Series(i)
        s.transform(func)
        end = timer()
        time.append(end - start)
    return time


def jit_performance(list_ndarray):
    @njit
    def sum(x):
        return x + 1
    @njit
    def square(x):
        return x ** 2
    @njit
    def transform(i):
        func = (sum, square)
        s = pd.Series(i)
        s.transform(func)
    time = []
    for i in list_ndarray:
        start = timer()
        transform(i)
        end = timer()
        time.append(end - start)
    return time


def get_random_ndarray(size):
    return np.random.randint(0, 100, size=size)


def get_performace_numbers():
    sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    avg_nj = []
    avg_j = []
    avg_jc = []
    p99_jc = []
    avg_first = []
    for size in sizes:
        list_ndarray = [get_random_ndarray(size) for _ in range(100)]
        time_not_jit = non_jit_performance(list_ndarray)
        time_jit = jit_performance(list_ndarray)
        iteration = [i+1 for i in range(0, 100)]
        plt.plot(iteration, time_not_jit, label=f'Non jitted', marker='.')
        plt.plot(iteration, time_jit, label=f'Jitted', marker='.')
        plt.xlabel("Iteration")
        plt.ylabel("Time in seconds")
        plt.yscale("log")
        plt.legend(loc='best')
        if os.path.isfile(f'perf_{size}.png'):
            os.remove(f'perf_{size}.png')
        plt.savefig(f'perf_{size}.png')
        plt.clf()
        print(f'stats for size {size}')
        avg_nj.append(sum(time_not_jit)/len(time_not_jit))
        print(f'avg non jit time : {sum(time_not_jit)/len(time_not_jit)}')
        avg_j.append(sum(time_jit) / len(time_jit))
        print(f'avg jit time : {sum(time_jit) / len(time_jit)}')
        avg_jc.append(sum(time_jit[1:]) / len(time_jit[1:]))
        print(f'avg jit time after cache : {sum(time_jit[1:]) / len(time_jit[1:])}')
        p99_jc.append(np.percentile(time_jit[1:], 99))
        print(f'n99 jit time after cache : {np.percentile(time_jit[1:], 99)}')
        avg_first.append(time_jit[0])
        print(f'Jit time for first query : {time_jit[0]}')
    plt.plot(sizes, avg_nj, label=f'Avg time - Non Jitted', marker='.')
    plt.plot(sizes, avg_j, label=f'Avg time - Jitted', marker='.')
    plt.plot(sizes, avg_jc, label=f'Avg time - Jitted w/o compilation overhead', marker='.')
    plt.plot(sizes, avg_first, label=f'Compilation overhead', marker='.')
    plt.plot(sizes, p99_jc, label=f'99th Percentile time - Jitted w/o compilation overhead', marker='.')
    plt.xlabel("Series sizes - number of row")
    plt.ylabel("Time in seconds")
    plt.yscale("log")
    plt.legend(loc='best')
    plt.savefig(f'scale.png')
    plt.clf()

get_performace_numbers()