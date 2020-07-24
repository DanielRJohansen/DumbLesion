import multiprocessing
import Constants
import time
import torch
from glob import glob
import os
import random

# Each process handles its own batch
# The thread makes sure the Batcher buffer is always full, and has acess to each process


class elem:
    def __init__(self, data, hist, label, name=None):
        self.data = data
        self.hist = hist
        self.label = label
        self.name = name

class Batcher:
    def __init__(self, folder, label_type, num_val_ims, cap_ims=False):
        self.train_folder = glob(folder + '\*')[0]
        self.val_folder = glob(folder + '\*')[1]
        self.num_agents = Constants.num_agents
        self.batch_size = Constants.batch_size
        self.s_size = Constants.section_depth
        self.mp_queue = multiprocessing.Queue()
        self.label_type = label_type

        self.batches_pr_val_round = int(num_val_ims/self.batch_size)
        self.train_lut = []
        self.val_lut = []
        self.val_batches = []

        self.cap_ims = cap_ims
        self.__makeLUT()
        self.__prepValBatches()
        self._val_batches = self.val_batches.copy()


        self.temp_batch = None



        self.workers = []
        for i in range(self.num_agents):
            pass
            worker = Worker(self.mp_queue, self.train_lut.copy())
            worker.start()
            self.workers.append(worker)
        print("Batcher initialized with batch_size {} and {} workers.".format(self.batch_size, self.num_agents))

    def cleanElement(self, string):
        index = 0
        while index < len(string):
            if string[index] == '_':
                break
            index += 1
        return string[:index]


    def __makeLUT(self):
        train_count = 0
        for sf in glob(self.train_folder+'\*'):
            elements = os.listdir(sf)
            elements = list(set(list(map(self.cleanElement, elements))))    # Stackoverflow, obviously..
            for e in elements:
                data = os.path.join(sf, e+"_section.pt")
                hist = os.path.join(sf, e+"_hist.pt")
                if self.label_type == "AoC":
                    label = os.path.join(sf, e+"_AOCLabel.pt")
                else:
                    label = os.path.join(sf, e+"_zlabel.pt")
                self.train_lut.append(elem(data, hist, label, name=sf+'/'+e))
            train_count += 1
            if train_count > 50 and self.cap_ims:
                break

        for sf in glob(self.val_folder + '\*'):
            elements = os.listdir(sf)
            elements = list(set(list(map(self.cleanElement, elements))))  # Stackoverflow, obviously..
            for e in elements:
                data = os.path.join(sf, e + "_section.pt")
                hist = os.path.join(sf, e + "_hist.pt")
                if self.label_type == "AoC":
                    label = os.path.join(sf, e + "_AOCLabel.pt")
                else:
                    label = os.path.join(sf, e + "_zlabel.pt")
                self.val_lut.append(elem(data, hist, label))


    def shutOff(self):
        for worker in self.workers:
            worker.terminate()


    def getBatch(self):     # Called from NN only

        ## Temproary
        #hile self.mp_queue.empty():
        #    time.sleep(0.001)
        #if self.temp_batch is None:
        #    batch = self.mp_queue.get()
        #    self.temp_batch = batch
        #batch = self.temp_batch
        #return batch.data, batch.hist, batch.label  # data, label

        ######
        while self.mp_queue.empty():
            time.sleep(0.001)
        batch = self.mp_queue.get()
        return batch.data, batch.hist, batch.label   # data, label

    def __prepValBatches(self):
        for i in range(self.batches_pr_val_round):
            batch = []
            hists = []
            labels = []

            for i in range(self.batch_size):
                section = self.val_lut.pop()
                data = torch.load(section.data)
                if data.shape != (7, 256, 256):
                    print(section.name)
                    exit()
                batch.append(data.unsqueeze(0))  # Unsqueeze adds channel dimension
                hists.append(torch.load(section.hist))
                labels.append(torch.load(section.label))

            batch = torch.stack(batch, dim=0)
            hists = torch.stack(hists, dim=0)
            labels = torch.stack(labels, dim=0)

            self.val_batches.append(elem(batch, hists, labels))

    def getValBatch(self):
        if len(self._val_batches) == 0:
            self._val_batches = self.val_batches.copy()
            return None, None, None
        batch = self._val_batches.pop()
        return batch.data, batch.hist, batch.label

class Worker(multiprocessing.Process):
    def __init__(self, queue, train_lut):
        multiprocessing.Process.__init__(self)
        self.queue = queue
        self.max_q_size = Constants.max_q_size
        self.batch_size = Constants.batch_size
        self.train_lut = train_lut
        #self.val_lut = val_lut


    def run(self):
        while True:
            batch = []
            hists = []
            labels = []
            for i in range(self.batch_size):
                section = random.choice(self.train_lut)
                data = torch.load(section.data)
                if data.shape != (7,256,256):
                    print(section.name)
                    exit()
                batch.append(data.unsqueeze(0))  # Unsqueeze adds channel dimension
                hists.append(torch.load(section.hist))
                labels.append(torch.load(section.label))

            batch = torch.stack(batch, dim=0)
            hists = torch.stack(hists, dim=0)
            labels = torch.stack(labels, dim=0)



            while self.queue.qsize() > Constants.max_batches_in_ram:
                pass
            self.queue.put(elem(batch, hists, labels))       # TODO load the images as arrays

