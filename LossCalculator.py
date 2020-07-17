import torch
import Constants

loss = torch.nn.CosineSimilarity()

class OrderLoss:
    def __init__(self):
        pass

    def calcLoss(self, predictions, labels):
        predictions = predictions.tolist()
        length = len(predictions[0])
        loss = torch.tensor([1], dtype=torch.float32)

        for prediction, label in zip(predictions, labels):

            order = []
            for i in range(length):
                m = min(prediction)
                for j in range(length):
                    if prediction[j] == m:
                        order.append(j)
                        prediction[j] = 2 # Just needs to be higher than 1
                        break
            dist = torch.dist(torch.tensor(order).type(torch.float32), label.type(torch.float32))
            loss = torch.add(loss, dist)

        loss = torch.div(loss, len(predictions))
        loss.requires_grad_()

        acc = 1-loss.item()
        return loss, acc


def zLoss(predictions, labels, device):
    loss = torch.mean(torch.abs(torch.sub(torch.flatten(predictions), labels)))
    acc = 1-loss.item()
    return loss, acc


def IoULoss(prediction, label, device):
    cap = torch.ones((label.shape[0], label.shape[1], label.shape[2])).to(device)
    intersect = torch.sum(torch.mul(prediction, label))
    union = torch.add(prediction, label)
    union = torch.where(union > 1, cap, union)
    union = torch.sum(union)

    if intersect.item() == 0:
        return torch.tensor(2)

    loss = torch.div(union, intersect)
    label_sum = torch.sum(label)
    acc = intersect.item()/label_sum - (union-label_sum)/(32*32)

    return loss, acc.item()


