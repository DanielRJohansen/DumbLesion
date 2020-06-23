class ScoreBoard:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.rights = [0] * num_classes
        self.wrongs = [0] * num_classes

    def getAcc(self):
        acc = []
        for r, w in zip(self.rights, self.wrongs):
            acc.append(r/(r+w))
        return acc

    def reset(self):
        for i in range(self.num_classes):
            self.rights[i] = 0
            self.wrongs[i] = 0