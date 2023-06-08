import multiprocessing
from utils.replay_memory import Memory
from utils.torch import *
# from utils.vista_helper import *
import math
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"

def worker_func(worker_id, pid, queue):
    print(f"Worker {worker_id}, PID: {pid}")
    queue.put(worker_id)

class Agent:

    def __init__(self, num_threads=1):
        self.num_threads = num_threads

    def collect_samples(self, min_batch_size, mean_action=False, render=False):
        t_start = time.time()
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []
        for i in range(self.num_threads-1):
            pid = multiprocessing.current_process().pid
            worker_args = (i+1, pid, queue)
            worker = multiprocessing.Process(target=worker_func, args=worker_args)
            workers.append(worker)

        for worker in workers:
            worker.start()
        # memory, log = collect_samples(0, None, self.env, self.policy, self.custom_reward, mean_action,
        #                               render, self.running_state, thread_batch_size)
        for _ in workers:
            worker_id = queue.get()
            print(f"worker id: {worker_id}")

        return None
