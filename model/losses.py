import torch
import torch.nn as nn
import torch.nn.functional as f


class WeightedBceLoss(nn.Module):
    """ Wrapped torch's binary cross entropy loss that can be weighted """
    def __init__(self):
        """"
        TODO: class weights
        """
        super().__init__()

    def forward(self, y_pred, y, weights=None):
        """
        Calculate the weighted BCE loss. If no weights are provided, normal BCE loss will be calculated.
    
        :param y_pred: prediction
        :param y: ground truth
        :param weights: weights
        :return: BCE loss averaged
        """
        out = f.binary_cross_entropy(y_pred, y, reduction='none')
        if weights is not None:
            out = out * weights
        loss = out.mean()
        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weights=None, smooth=1):
        # have to use contiguous since they may from a torch.view op
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)

        return (1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))).mean()


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = WeightedBceLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target, weights=None):
        bce = self.bce(pred, target, weights)
        dice = self.dice(pred, target, weights)
        return self.bce_weight * bce + self.dice_weight * dice

