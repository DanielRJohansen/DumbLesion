import torch

loss = torch.nn.CosineSimilarity()

class OrderLoss:
    def __init__(self):
        pass

    def calcLoss(self, predictions, labels):
        predictions = predictions.tolist()
        orders = []

        for prediction, label in zip(predictions, labels):
            length = len(prediction)
            order = []
            for i in range(length):
                m = min(prediction)
                for j in range(length):
                    if prediction[j] == m:
                        order.append(j)
                        prediction[j] = 9999999
                        break
            orders.append(order)
        orders = torch.tensor(orders, dtype=torch.float32)
        labels = labels.type(torch.float32)
        orders = torch.add(orders, 1)
        labels = torch.add(labels, 1)
        batch_loss = torch.mean(loss(orders, labels))
        batch_loss.requires_grad_()
        return batch_loss
        #print(orders)
        #print(labels)
        #print(loss(orders, labels))




def IoULoss():
    pass