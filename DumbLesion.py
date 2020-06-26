import torch
import torch.nn as nn
import Stats

class DLCNN(nn.Module):
    def __init__(self, device, lr):
        super(DLCNN, self).__init__()
        self.device = device
        self.lr = lr

        self.conv1_pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(3, 256, 3)  # In_channels, outchannelse, kernelsize
        self.bn1 = nn.BatchNorm2d(256)  # Batch normalization layer
        self.conv1_relu = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 128, 3)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(64, 32, 3)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, 3)
        self.bn5 = nn.BatchNorm2d(32)

    def __forward(self, batch):
        batch = self.conv1_pad(batch)
        print(batch.shape)



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
    def __init__(self, batch_size, test_batch_size, lr, epochs, sample_size, trainfolder, testfolder, batcher, bpe):
        # Train stats
        self.epochs = epochs
        self.batch_size = batch_size
        self.acc_history = []
        self.loss_history = []
        self.device = torch.device('cuda:0')

        # Batcher
        self.train_folder = trainfolder
        self.test_folder = testfolder
        self.batcher = batcher
        self.batches_per_epoch = bpe
        print("Training for {} epochs, with {} episodes pr epoch.".format(epochs, self.batches_per_epoch))

        # Pretrained CNN
        self.CNN = DLCNN()
        #self.CNN.eval()     # Really crucial!
        self.CNN.to(self.device)


        # Classifier
        self.Scoreboard = ScoreBoard(self.num_classes)
        self.classifier = Top(input_nodes=outputsize, device=self.device, lr=lr, num_classes=self.num_classes, scoreboard=self.Scoreboard,  MLP=mlp)
        self.inputs = []
        self.labels = []

    def orderLoss(self):
        pass

    def IoULoss(self):
        pass