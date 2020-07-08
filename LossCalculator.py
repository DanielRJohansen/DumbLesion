import torch

def orderLoss(prediction, label):
    print(prediction)
    length = list(prediction.size())[0]
    print(length)
    order = []
    for i in range(length):
        m = torch.min(prediction).item()
        for j in range(length):
            if prediction[j] == m:
                order.append(j)
                prediction[j] = 9999999
                break

    print(order)
    print(label)



def IoULoss():
    pass