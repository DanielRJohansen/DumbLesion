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
        self.three = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)

        self.five_bottle = nn.Conv3d(B_input_channels, 8, kernel_size=1, stride=1)
        self.five = nn.Conv3d(8, 8, kernel_size=5, stride=1, padding=2)

        self.flat_bottle = nn.Conv3d(B_input_channels, 8, kernel_size=1, stride=1)
        self.flat = nn.Conv3d(8, 8, kernel_size=(1,3,3), stride=1, padding=(0,1,1))

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        self.maxpool_bottle = nn.Conv3d(B_input_channels, 8, kernel_size=1, stride=1)

        self.bn = nn.BatchNorm3d(64)

        print("Block weights:", sum(p.numel() for p in self.parameters() if p.requires_grad))


    def forward(self, batch):
        one = self.one(batch)
        three = self.three_bottle(batch)
        three = self.three(three)
        five = self.five_bottle(batch)
        five = self.five(five)
        flat = self.flat_bottle(batch)
        flat = self.flat(flat)
        max = self.maxpool(batch)
        max = self.maxpool_bottle(max)
        batch = torch.cat((one, three, five, flat, max), dim=1)  # dim 0 is section nr in batch, 1 is channel
        batch = self.bn(batch)
        return batch

class DLCNN(nn.Module):
    def __init__(self, block_B_size=1):
        super(DLCNN, self).__init__()


        first_out = 64
        self.intro_conv = nn.Conv3d(1, first_out, kernel_size=3, stride=(1,2,2),padding=1)  # To keep depth of 7
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




        # Block B
        self.B_adapter_bottle = nn.Conv3d(128, 64, kernel_size=1, stride=1)
        self.B_adapter_depth = nn.Conv3d(64, 64, kernel_size=(5,5,5), stride=1, padding=(0,2,2))
        self.B_blocks = nn.ModuleList()
        for i in range(block_B_size):
            self.B_blocks.append(BlockB())


        # Adapter out
        self.out_maxpool = nn.MaxPool3d(kernel_size=3, stride=(1,2,2), padding=1)   # 64x7x64x64
        self.out_bottle = nn.Conv3d(64,16,kernel_size=1, stride=1)                  # 16x7x64x64
        self.out_conv = nn.Conv3d(16, 16, kernel_size=3, stride=(1,2,2), padding=1)  # 8x7x32x32
        self.out_bn = nn.BatchNorm3d(16)

        self.weigths = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("BaseCNN loaded. Weights: ", f"{self.weigths:,}")

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


        # Block B (multiple)
        batch = self.B_adapter_bottle(batch)
        batch = self.B_adapter_depth(batch)
        for block in self.B_blocks:
            batch = block.forward(batch)

        # Adapter for linear output
        batch = self.out_maxpool(batch)
        batch = self.out_bottle(batch)
        batch = self.out_conv(batch)
        batch = self.out_bn(batch)
        #batch = batch.view(batch.size()[0], -1)
        #batch = self.out_layer(batch)
        #batch = torch.cat((branch_1, batch), 1)
        return batch

    def loadModel(self):
        self.load_state_dict(torch.load(Constants.model_path))

    def saveModel(self):
        torch.save(self.state_dict(), Constants.model_path)




class zTop(nn.Module):
    def __init__(self):
        super(zTop, self).__init__()
        self.flat = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=(0,1,1))
        self.bn = nn.BatchNorm3d(16)      # Only works with batch_size > 1
        self.fc = nn.Linear(16*1*32*32, 1)
        self.sm = nn.Softmax(1)

        self.weigths = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("zTop loaded. Weights: ", f"{self.weigths:,}")


    def forward(self, batch):
        batch = self.flat(batch)
        batch = self.bn(batch)
        batch = batch.view(batch.size()[0], -1)
        batch = self.fc(batch)
        batch = self.sm(batch)
        return batch


class AoCTop(nn.Module):
    def __init__(self):
        super(AoCTop, self).__init__()
        self.flat = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=(0,1,1))
        self.bottle = nn.Conv3d(16, 1, kernel_size=1, stride=1, padding=0)
        self.af = nn.Sigmoid()

        self.weigths = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("AoC Top loaded. Weights: ", f"{self.weigths:,}")

    def forward(self, batch):
        batch = self.flat(batch)
        batch = self.bottle(batch)
        batch = torch.squeeze(batch)                    # Removes depth and channel!
        batch = self.af(batch)
        return batch