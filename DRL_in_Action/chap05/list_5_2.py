from re import I
import numpy as np
import multiprocessing as mp


def square(i, x, queue):
    print(f"In process {i}")
    queue.put(np.square(x))


if __name__ == '__main__':

    processes = []
    queue = mp.Queue()
    x = np.arange(64)
    for i in range(8):
        start_index = 8*i
        proc = mp.Process(target=square, args=(
            i, x[start_index:start_index+8], queue))
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

    for proc in processes:
        proc.terminate()

    results = []
    while not queue.empty():
        results.append(queue.get())

    print(results)
