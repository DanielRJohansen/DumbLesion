import torch

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
        return loss

def zLoss(predictions, labels):
    return torch.mean(torch.abs(torch.sub(torch.flatten(predictions), labels)))



def IoULoss():
    pass