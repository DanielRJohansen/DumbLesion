import torch

loss = torch.nn.CosineSimilarity()

class OrderLoss:
    def __init__(self):
        pass

    def calcLoss(self, predictions, labels):
        predictions = predictions.tolist()
        length = len(predictions[0])
        loss = torch.zeros(length)

        for prediction, label in zip(predictions, labels):

            order = []
            for i in range(length):
                m = min(prediction)
                for j in range(length):
                    if prediction[j] == m:
                        order.append(j)
                        prediction[j] = 2 # Just needs to be higher than 1
                        break
            dif = torch.sub(labels, torch.tensor(order))
            loss = torch.add(loss, dif)
            #print(label, order)

        loss = torch.div(loss, len(predictions))
        loss.requires_grad_()
        print(loss)
        return loss
        #orders = torch.tensor(orders, dtype=torch.float32)
        #labels = labels.type(torch.float32)
        #orders = torch.add(orders, 1)
        #labels = torch.add(labels, 1)
        #batch_loss = torch.mean(loss(orders, labels))
        #batch_loss = 1 /batch_loss
        #batch_loss.requires_grad_()
        #return batch_loss
        #print(orders)
        #print(labels)
        #print(loss(orders, labels))




def IoULoss():
    pass