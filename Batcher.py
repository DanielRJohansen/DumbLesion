import multiprocessing
from Slicer import Slicer


# Each process handles its own batch
# The thread makes sure the Batcher buffer is always full, and has acess to each process
class Batcher:
    def __init__(self, workfolder,  LUT, num_agents, batch_size):
        self.work_folder = workfolder
        self.LUT = LUT
        self.num_agents = num_agents
        self.batch_size = batch_size

        self.buffer = [None] * num_agents       # Batch per agent
        self.lock = [False] * num_agents        # Batch is ready
        print(self.buffer)
        print(self.lock)
        for i in range(num_agents):
            agent = multiprocessing.Process(target=self.work, args=[i])
            agent.start()
        print("Batcher initialized")

    def work(self, agent_index):
        slicer = Slicer(self.work_folder, self.LUT)
        while True:
            batch = []
            for i in range(self.batch_size):
                batch.append(slicer.getSection())

            while self.lock[agent_index]:
                pass

            self.buffer[agent_index] = batch
            self.lock[agent_index] = True
            print("got here")
            print(self.lock)

    def getBatch(self):
        while True:
            for i in range(self.num_agents):
                print(self.lock)
                if self.lock[i]:
                    batch = self.buffer[i].copy()
                    self.lock[i] = False
                    return batch





