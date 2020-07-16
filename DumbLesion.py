import torch
import torch.nn as nn
import torch.nn.functional as F
from DLmodules import DLCNN, zTop, AoCTop
import torch.optim as optim
import Stats
from BatcherV2 import Batcher
import Constants
import sys
import time
from LossCalculator import IoULoss, OrderLoss, zLoss
import numpy as np
import matplotlib.pyplot as plt


class DumbLesionNet(nn.Module):
    def __init__(self, output_type, num_val_ims):
        super(DumbLesionNet, self).__init__()
        self.output_type = output_type

        # Train stats
        self.epochs = Constants.epochs
        self.batch_size = Constants.batch_size
        self.batches_per_epoch = Constants.batches_per_epoch
        self.epoch_length = int(self.batches_per_epoch/self.batch_size)

        self.lr = Constants.lr
        self.device = torch.device('cuda:0')

        self.acc_history = []
        self.loss_history = []
        self.val_acc_history = []

        # Batcher
        self.batcher = Batcher(Constants.work_folder, label_type=self.output_type, num_val_ims=num_val_ims)
        print("Training for {} epochs, with {} episodes pr epoch.".format(Constants.epochs, self.batches_per_epoch))

        # CNN + TOP modules
        self.base = DLCNN(block_B_size=4).to(self.device)
        if self.output_type == "AoC":
            self.top = AoCTop().to(self.device)
        elif self.output_type == "z":
            self.top = zTop().to(self.device)
        self.weigths = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Dumblesion loaded. Weights: ", f"{self.weigths:,}")
        print()


        # Utilities
        if self.output_type == "AoC":
            self.loss = IoULoss
        else:
            self.loss = zLoss
        #self.loss = IoULoss if self.output_type == "AoC" else self.loss = zLoss

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def __forward(self, batch):
        batch = batch.to(self.device)
        batch = self.base.forward(batch)
        batch = self.top.forward(batch)
        return batch

    def _train(self):
        best_acc = -999999

        for i in range(self.epochs):
            self.train()  # Notify PyTorch entering training mode.
            ep_loss = 0
            ep_acc = []
            e_time = time.time()

            for j in range(self.epoch_length):  # Input: batchsize, kernels, width, height
                self.optimizer.zero_grad()

                input, hist, label = self.batcher.getBatch()
                #input.require_grad = True
                prediction = self.__forward(input)

                loss, acc = self.loss(prediction, label.to(self.device), self.device)

                ep_acc.append(acc)
                ep_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            # Validation
            self.val_acc_history.append(self.validate())
            if self.val_acc_history[-1] > best_acc:
                best_acc = self.val_acc_history[-1]
                self.base.saveModel()

            self.acc_history.append(np.mean(ep_acc))
            self.loss_history.append(np.mean(ep_loss))
            e_time = time.time() - e_time
            print("Finish epoch {}. Epoch loss: {:.2f}. Train accuracy: {}. Val accuracy: {}. Time: {}.".format(
                i, ep_loss/self.epoch_length, np.mean(ep_acc), self.val_acc_history[-1], e_time))

        self.plotStuffs(best_acc)
        self.batcher.shutOff()

    def validate(self):
        self.eval()
        acc = 0.
        count = 0
        while True:
            data, hist, label = self.batcher.getValBatch()
            if data is None:
                break

            prediction = self.__forward(data)
            loss, _acc = self.loss(prediction, label.to(self.device), self.device)
            acc += _acc
            count += 1
            del prediction, loss

        return acc/count





    def plotStuffs(self, best_acc):
        fig, axs = plt.subplots(2)
        #plt.plot(self.loss_history, 'b', label="Loss")
        #plt.legend()

        axs[0].plot(self.acc_history, label="Train acc.")
        axs[0].plot(self.val_acc_history, label="Val Acc.")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        axs[0].legend()

        axs[1].plot(self.loss_history, label="Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        axs[1].legend()
        #ax2 = ax.twinx()
        #plt.plot(self.acc_history, 'r', label="Train acc.")
        #plt.legend()

        #p2 = plt.plot(self.val_acc_history, label="Val Acc.")
        #plt.legend()

        plt.savefig('./Plots/{:.4f}_plot.png'.format(best_acc))
        plt.show()

        #fig, (ax1, ax2) = plt.subplots(1, 2)
        #fig.suptitle('Horizontally stacked subplots')
        #.plot(x, y)
        #ax2.plot(x, -y)

    def trainMode(self):
        self.train()

    def inferenceMode(self):
        self.eval()