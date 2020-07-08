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
        self.one = nn.Conv3d(B_input_channels, 8, kernel_size=1, stride=1)

        self.three_bottle = nn.Conv3d(B_input_channels, 16, kernel_size=1, stride=1)
        self.three = nn.Conv3d(16, 24, kernel_size=3, stride=1, padding=1)

        self.five_bottle = nn.Conv3d(B_input_channels, 16, kernel_size=1, stride=1)
        self.five = nn.Conv3d(16, 16, kernel_size=5, stride=1, padding=2)

        self.seven_bottle = nn.Conv3d(B_input_channels, 16, kernel_size=1, stride=1)
        self.seven = nn.Conv3d(16, 8, kernel_size=7, stride=1, padding=3)

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        self.maxpool_bottle = nn.Conv3d(B_input_channels, 8, kernel_size=1, stride=1)

        self.bn = nn.BatchNorm3d(64)


    def forward(self, batch):
        one = self.one(batch)
        three = self.three_bottle(batch)
        three = self.three(three)
        five = self.five_bottle(batch)
        five = self.five(five)
        seven = self.seven_bottle(batch)
        seven = self.seven(seven)
        max = self.maxpool(batch)
        max = self.maxpool_bottle(max)
        batch = torch.cat((one, three, five, seven, max), dim=1)  # dim 0 is section nr in batch, 1 is channel
        batch = self.bn(batch)
        return batch

class DLCNN(nn.Module):
    def __init__(self, block_B_size=1):
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
        self.Branch1_bottle = nn.Conv3d(128, 16, kernel_size=1, stride=1)
        self.Branch1_conv = nn.Conv3d(16, 16, kernel_size=7, stride=(1, 4, 4), padding=3)
        self.Branch1_bottle2 = nn.Conv3d(16, 4, kernel_size=1, stride=1)
        self.Branch1_conv2 = nn.Conv3d(4, 4, kernel_size=7, stride=(1, 4, 4), padding=3)
        self.Branch1_fc = nn.Linear(int(4*7*16*16), 20)


        # Block B
        self.B_adapter = nn.Conv3d(128, 64, kernel_size=1, stride=1)
        self.B_blocks = nn.ModuleList()
        for i in range(block_B_size):
            self.B_blocks.append(BlockB())


        self.out_maxpool = nn.MaxPool3d(kernel_size=3, stride=(1,2,2), padding=1)   # 64x7x128x128
        self.out_bottle = nn.Conv3d(64,16,kernel_size=1, stride=1)                  # 16x7x128x128
        self.out_conv = nn.Conv3d(16, 8, kernel_size=3, stride=(1,2,2), padding=1)  # 8x7x64x64
        self.out_conv2 = nn.Conv3d(8,8,kernel_size=3, stride=(1,2,2), padding=1)    # 8x7x32x32
        self.out_layer = nn.Linear(int(8*7*32*32), 980)

    def forward(self, batch):
        batch = self.intro_conv(batch)
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
        del one, three, five, seven, max

        branch_1 = self.Branch1_bottle(batch)
        branch_1 = self.Branch1_conv(branch_1)
        branch_1 = self.Branch1_bottle2(branch_1)
        branch_1 = self.Branch1_conv2(branch_1)
        branch_1 = branch_1.view(branch_1.size()[0], -1)
        branch_1 = self.Branch1_fc(branch_1)


        # Block B (multiple)
        batch = self.B_adapter(batch)
        for block in self.B_blocks:
            batch = block.forward(batch)

        # Adapter for linear output
        batch = self.out_maxpool(batch)
        batch = self.out_bottle(batch)
        batch = self.out_conv(batch)
        batch = self.out_conv2(batch)
        batch = batch.view(batch.size()[0], -1)
        batch = self.out_layer(batch)
        batch = torch.cat((branch_1, batch), 1)
        return batch





class Top(nn.Module):
    def __init__(self, input_nodes, output_nodes, MLP=False):
        super(Top, self).__init__()
        self.MLP = MLP

        if MLP:
            self.fc_in = nn.Linear(input_nodes, 256)
            self.fc1 = nn.Linear(256, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc_out = nn.Linear(128, output_nodes)
        else:
            self.fc_single = nn.Linear(input_nodes, output_nodes)
        self.bn = nn.BatchNorm1d(output_nodes)      # Only works with batch_size > 1

        if MLP:
            print("MLP top loaded")
        else:
            print("SLP top loaded")


    def forward(self, batch_data):
        if self.MLP:
            batch_data = self.fc_in(batch_data)
            batch_data = self.fc1(batch_data)
            batch_data = self.fc2(batch_data)
            batch_data = self.fc_out(batch_data)
        else:
            batch_data = self.fc_single(batch_data)
        #print(batch_data.shape)
        #batch_data = self.bn(batch_data)
        return batch_data