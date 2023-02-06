import multiprocessing as mp
import numpy as np


def square(x):
    return np.square


if __name__ == '__main__':
    x = np.arange(64)

    pool = mp.Pool(mp.cpu_count())

    squared = pool.map(square, [x[8*i:8*i+8] for i in range(8)])

    print(squared)
