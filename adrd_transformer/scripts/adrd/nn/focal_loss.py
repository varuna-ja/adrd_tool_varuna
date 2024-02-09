import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidFocalLoss(nn.Module):
    ''' ... '''
    def __init__(
        self,
        beta: float = 0.9999,
        gamma: float = 2.0,
        scale: float = 1.0,
        num_per_cls = (1, 1),
        reduction: str = 'mean',
    ):
        ''' ... '''
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        # weights to balance loss
        weight_neg = (1 - beta) / (1 - beta ** num_per_cls[0])
        weight_pos = (1 - beta) / (1 - beta ** num_per_cls[1])
        self.weight_neg = weight_neg * scale
        self.weight_pos = weight_pos * scale
        
    def forward(self, input, target):
        ''' ... '''
        p = torch.sigmoid(input)
        p_t = p * target + (1 - p) * (1 - target)
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        alpha_t = self.weight_pos * target + self.weight_neg * (1 - target)
        loss = alpha_t * loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss