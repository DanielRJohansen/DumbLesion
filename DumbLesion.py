import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Stats
import Constants
from Batcher import Batcher
import sys
import time
B_input_channels = 64

class BlockB(nn.Module):
    def __init__(self):
        super(BlockB, self).__init__()
        self.one = nn.Conv3d(B_input_channels, 16, kernel_size=1, stride=1)

        self.three_bottle = nn.Conv3d(B_input_channels, 16, kernel_size=1, stride=1)
        self.three = nn.Conv3d(16, 48, kernel_size=3, stride=1, padding=1)

        self.five_bottle = nn.Conv3d(B_input_channels, 16, kernel_size=1, stride=1)
        self.five = nn.Conv3d(16, 32, kernel_size=5, stride=1, padding=2)

        self.seven_bottle = nn.Conv3d(B_input_channels, 16, kernel_size=1, stride=1)
        self.seven = nn.Conv3d(16, 16, kernel_size=7, stride=1, padding=3)

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        self.maxpool_bottle = nn.Conv3d(B_input_channels, 16, kernel_size=1, stride=1)

    def forward(self, batch):
        one = self.one(batch)
        three = self.three_bottle(batch)
        three = self.three(three)
        five = self.five_bottle(batch)
        five = self.five(five)
        seven = self.B1_7_bottle(batch)
        seven = self.B1_7(seven)
        max = self.B1_maxpool(batch)
        max = self.B1_maxpool_bottle(max)
        batch = torch.cat((one, three, five, seven, max), dim=1)  # dim 0 is section nr in batch, 1 is channel
        batch = self.B1_bn(batch)

class DLCNN(nn.Module):
    def __init__(self, block_B_size=5):
        super(DLCNN, self).__init__()


        first_out = 64
        self.CNN = nn.Sequential()
        self.intro_conv = nn.Conv3d(1, first_out, kernel_size=3, stride=(1,2,2),padding=1)  # To keep depth of 7
        #self.intro_conv = nn.Conv3d(1, first_out, kernel_size=3, stride=(2,2,2), padding=(0,1,1))  # F0r depth 3
        self.intro_bn = nn.BatchNorm3d(first_out)
        self.intro_relu = nn.ReLU()     # Remove?


        # Block A
        self.B1_1 = nn.Conv3d(first_out, 16, kernel_size=1, stride=1)

        self.B1_3_bottle = nn.Conv3d(first_out, 16, kernel_size=1, stride=1)
        self.B1_3 = nn.Conv3d(16, 48, kernel_size=3, stride=1, padding=1)

        self.B1_5_bottle = nn.Conv3d(first_out, 16, kernel_size=1, stride=1)
        self.B1_5 = nn.Conv3d(16, 32, kernel_size=5, stride=1, padding=2)

        self.B1_7_bottle = nn.Conv3d(first_out, 16, kernel_size=1, stride=1)
        self.B1_7 = nn.Conv3d(16, 16, kernel_size=7, stride=1, padding=3)

        self.B1_maxpool = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        self.B1_maxpool_bottle = nn.Conv3d(first_out, 16, kernel_size=1, stride=1)

        self.B1_bn = nn.BatchNorm3d(128)
        # TODO Relu here?

        # Branch 1      - might indicate body area
        conv_dims = int(16*7*16*16)     #channels, depth, y, x
        self.Branch1_bottle = nn.Conv3d(128, 16, kernel_size=1, stride=1)
        self.Branch1_conv = nn.Conv3d(16, 16, kernel_size=7, stride=(1, 4, 4), padding=3)
        self.Branch1_conv2 = nn.Conv3d(16, 16, kernel_size=7, stride=(1, 4, 4), padding=3)
        self.Branch1_fc = nn.Linear(conv_dims, 20)


        # Block B
        B_blocks = []

        self.teststuff()

    def teststuff(self):
        pass

    def forward(self, batch):
        print("Input batch", batch.shape)
        batch = self.intro_conv(batch)
        print(batch.shape)
        batch = self.intro_bn(batch)
        batch = self.intro_relu(batch)

        # Block 1
        one = self.B1_1(batch)
        three = self.B1_3_bottle(batch)
        three = self.B1_3(three)
        five = self.B1_5_bottle(batch)
        five = self.B1_5(five)
        seven = self.B1_7_bottle(batch)
        seven = self.B1_7(seven)
        max = self.B1_maxpool(batch)
        max = self.B1_maxpool_bottle(max)
        batch = torch.cat((one, three, five, seven, max), dim=1)    # dim 0 is section nr in batch, 1 is channel
        batch = self.B1_bn(batch)

        branch_1 = self.Branch1_bottle(batch)
        branch_1 = self.Branch1_conv(branch_1)
        branch_1 = self.Branch1_conv2(branch_1)
        print(branch_1.shape)
        branch_1 = branch_1.view(branch_1.size()[0], -1)
        branch_1 = self.Branch1_fc(branch_1)

        print("Result: ", branch_1)






class Top(nn.Module):
    def __init__(self, input_nodes, device, lr, num_classes, scoreboard, MLP=False):
        super(Top, self).__init__()
        self.device = device
        self.lr = lr
        self.num_classes = num_classes
        self.MLP = MLP

        self.fc_single = nn.Linear(input_nodes, self.num_classes)

        self.fc_in = nn.Linear(input_nodes, 1000)
        self.fc1 = nn.Linear(1000, 5000)
        self.fc2 = nn.Linear(5000, 5000)
        self.fc3 = nn.Linear(5000, 200)
        self.fc_out = nn.Linear(200, self.num_classes)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.scoreboard = scoreboard

        self.to(self.device)

        if MLP:
            print("MLP classifier loaded")
        else:
            print("SLP classifier loaded")

    def singleForward(self, input):
        prediction = self.__forward(input)
        prediction = F.softmax(prediction, dim=1)
        prediction = torch.argmax(prediction, dim=1)
        return prediction

    def __forward(self, batch_data):
        if self.MLP:
            batch_data = self.fc_in(batch_data)
            batch_data = self.fc1(batch_data)
            batch_data = self.fc2(batch_data)
            batch_data = self.fc3(batch_data)
            batch_data = self.fc_out(batch_data)
        else:
            batch_data = self.fc_single(batch_data)

        return batch_data

    def trainMode(self):
        self.train()

    def evalMode(self):
        self.eval()

    def fit(self, inputs, labels):
        self.optimizer.zero_grad()

        prediction = self.__forward(inputs)
        prediction = F.softmax(prediction, dim=1)

        loss = self.loss(prediction, labels)
        loss.backward()
        self.optimizer.step()

    def batchAccuracy(self, inputs, labels):
        prediction = self.__forward(inputs)
        prediction = F.softmax(prediction, dim=1)
        classes = torch.argmax(prediction, dim=1)

        for i in range(self.num_classes):
            animal_tensor = torch.ones(classes.size(), dtype=torch.int8).to(self.device)
            animal_tensor *= i
            animal_rights = torch.where((classes == labels) & (classes == animal_tensor), torch.tensor([1.]).to(self.device), torch.tensor([0.]).to(self.device))
            animal_wrongs = torch.where((classes != labels) & (labels == animal_tensor), torch.tensor([1.]).to(self.device), torch.tensor([0.]).to(self.device))
            self.scoreboard.rights[i] = torch.sum(animal_rights).item()
            self.scoreboard.wrongs[i] = torch.sum(animal_wrongs).item()


class DumbLesionNet:
    def __init__(self, batcher):
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

        # Pretrained CNN
        self.CNN = DLCNN(device=self.device)
        #self.CNN.eval()     # Really crucial!
        self.CNN.to(self.device)


        # Classifier
        #self.Scoreboard = ScoreBoard(self.num_classes)
        #self.classifier = Top(input_nodes=outputsize, device=self.device, lr=lr, num_classes=self.num_classes, scoreboard=self.Scoreboard,  MLP=mlp)
        self.inputs = []
        self.labels = []

    def go(self):
        data, labels = self.batcher.getBatch()
        print("Ready to send")

        stuff = self.CNN.forward(data.to(self.device))

    def orderLoss(self):
        pass

    def IoULoss(self):
        pass