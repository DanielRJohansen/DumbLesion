import torch
import torch.nn as nn
import torch.nn.functional as F
from DLmodules import DLCNN, zTop
import torch.optim as optim
import Stats
import Constants
import sys
import time
from LossCalculator import IoULoss, OrderLoss, zLoss
import numpy as np


class DumbLesionNet(nn.Module):
    def __init__(self, batcher):
        super(DumbLesionNet, self).__init__()

        # Train stats
        self.epochs = Constants.epochs
        self.batch_size = Constants.batch_size
        self.batches_per_epoch = Constants.batches_per_epoch
        self.epoch_length = int(self.batches_per_epoch/self.batch_size)

        self.lr = Constants.lr
        self.device = torch.device('cuda:0')

        self.acc_history = []
        self.loss_history = []

        # Batcher
        self.batcher = batcher
        print("Training for {} epochs, with {} episodes pr epoch.".format(Constants.epochs, self.batches_per_epoch))

        # CNN + TOP modules
        self.model = nn.ModuleList((DLCNN(out_nodes=500, block_B_size=4), zTop()))
        self.model = self.model.to(self.device)
        self.weigths = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Dumblesion loaded. Weights: ", f"{self.weigths:,}")
        print()

        # Utilities
        self.loss = OrderLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def __forward(self, batch):
        batch = batch.to(self.device)
        for module in self.model:
            batch = module.forward(batch)
        return batch

    def _train(self):
        self.train()  # Notify PyTorch entering training mode.

        for i in range(self.epochs):
            ep_loss = 0
            ep_acc = []
            e_time = time.time()

            for j in range(self.epoch_length):  # Input: batchsize, kernels, width, height
                self.optimizer.zero_grad()

                input, hist, label = self.batcher.getBatch()
                input.require_grad = True
                label = label.to(self.device)
                prediction = self.__forward(input)
                prediction = F.softmax(prediction, dim=1)

                loss = zLoss(prediction, label)
                acc = torch.div(torch.tensor(1), loss)

                ep_acc.append(acc.item())
                self.acc_history.append(acc.item())
                ep_loss += loss.item()

                loss.backward()
                #print(prediction.grad)
                self.optimizer.step()
            e_time = time.time() - e_time
            print("Finish epoch {}. Epoch loss: {:.2f}. Train accuracy: {}. Time: {}.".format(
                i, ep_loss/self.epoch_length, np.mean(ep_acc),e_time))


    def go(self):
        data, labels = self.batcher.getBatch()
        print("Ready to send")
        stuff = self.__forward(data.to(self.device))
        print("stuff", stuff)

        a = torch.tensor([[2,1,0],[1,0,2],[0,1,2]])
        b = torch.tensor([[0,1,2],[0,1,2],[0,1,2]])
        #orderLoss(a, b)
        #orderLoss(stuff, labels)
