import torch
import torch.nn as nn
import torch.nn.functional as F
from DLmodules import DLCNN, Top
import torch.optim as optim
import Stats
import Constants
import sys
import time
from LossCalculator import IoULoss, orderLoss



class DumbLesionNet(nn.Module):
    def __init__(self, batcher):
        super(DumbLesionNet, self).__init__()

        # Train stats
        self.epochs = Constants.epochs
        self.batch_size = Constants.batch_size
        self.acc_history = []
        self.loss_history = []
        self.device = torch.device('cuda:0')

        # Batcher
        self.batcher = batcher
        self.batches_per_epoch = Constants.batches_per_epoch
        print("Training for {} epochs, with {} episodes pr epoch.".format(Constants.epochs, self.batches_per_epoch))

        # CNN + TOP modules
        self.model = nn.ModuleList((DLCNN(), Top(input_nodes=1000, output_nodes=7)))
        self.model = self.model.to(self.device)
        self.weigths = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Dumblesion loaded. Weights: ", f"{self.weigths:,}")

    def __forward(self, batch):
        for module in self.model:
            batch = module.forward(batch)
        return batch

    def go(self):
        data, labels = self.batcher.getBatch()
        print("Ready to send")
        stuff = self.__forward(data.to(self.device))
        print("stuff", stuff)
        orderLoss(stuff, labels)
