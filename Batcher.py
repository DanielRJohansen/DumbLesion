import multiprocessing
from Slicer import Slicer
import Constants
import time
import torch
# Each process handles its own batch
# The thread makes sure the Batcher buffer is always full, and has acess to each process


class Batcher:
    def __init__(self, LUT):
        self.work_folder = Constants.work_folder
        self.num_agents = Constants.num_agents
        self.batch_size = Constants.batch_size
        self.s_size = Constants.section_depth
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
        return batch[0], batch[1]   # data, label



class Worker(multiprocessing.Process):
    def __init__(self, queue, sections_lut):
        multiprocessing.Process.__init__(self)
        self.queue = queue
        self.max_q_size = Constants.max_q_size
        self.batch_size = Constants.batch_size
        self.slicer = Slicer(sections_lut)

    def run(self):
        while True:
            batch = []
            labels = []
            for i in range(self.batch_size):
                data, label = self.slicer.getSection()
                batch.append(data.unsqueeze(0))  # Unsqueeze adds channel dimension
                labels.append(label)

            batch = torch.stack(batch, dim=0)
            labels = torch.stack(labels, dim=0)
            while self.queue.qsize() > 2:
                pass
            self.queue.put((batch, labels))       # TODO load the images as arrays

