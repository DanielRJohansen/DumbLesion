import multiprocessing
from Slicer import Slicer
import Constants
import time

# Each process handles its own batch
# The thread makes sure the Batcher buffer is always full, and has acess to each process


class Batcher:
    def __init__(self, LUT):
        self.work_folder = Constants.work_folder
        self.num_agents = Constants.num_agents
        self.batch_size = Constants.batch_size
        self.s_size = Constants.section_size
        self.mp_queue = multiprocessing.Queue()

        for i in range(self.num_agents):
            worker = Worker(self.mp_queue, LUT)
            worker.start()
        print("Batcher initialized with batch_size {} and {} workers".format(self.batch_size, self.num_agents))

    def shutOff(self):
        pass


    def getBatch(self):     # Called from NN only
        while self.mp_queue.empty():
            time.sleep(0.001)
        batch = self.mp_queue.get()
        return batch



class Worker(multiprocessing.Process):
    def __init__(self, queue, sections_lut):
        multiprocessing.Process.__init__(self)
        self.queue = queue
        self.max_q_size = Constants.max_q_size
        self.batch_size = Constants.batch_size
        self.slicer = Slicer(sections_lut)

    def run(self):
        while True:
            t0 = time.time()
            batch = []
            for i in range(self.batch_size):
                batch.append(self.slicer.getSection())


            self.queue.put(batch)       # TODO load the images as arrays
