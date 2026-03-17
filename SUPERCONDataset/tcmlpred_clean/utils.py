import torch


def classification_accuracy(pred, target):

    pred_label = torch.argmax(pred, dim=1)

    correct = (pred_label == target).sum().item()

    return correct / len(target)