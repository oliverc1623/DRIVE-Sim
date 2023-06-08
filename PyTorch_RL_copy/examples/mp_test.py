import multiprocessing

def worker_func(worker_id, pid, queue):
    # print(f"Worker {worker_id}, PID: {pid}")
    queue.put(worker_id)

def run_multiprocessing():
    num_threads = 4
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    workers = []

    for i in range(num_threads-1):
        pid = multiprocessing.current_process().pid
        worker_args = (i+1, pid, queue)
        worker = multiprocessing.Process(target=worker_func, args=worker_args)
        workers.append(worker)
        
    for worker in workers:
        worker.start()

    for _ in workers:
        worker_id = queue.get()
        print(worker_id)

if __name__ == "__main__":
    for _ in range(5):  # Execute the code in a loop 5 times
        run_multiprocessing()
