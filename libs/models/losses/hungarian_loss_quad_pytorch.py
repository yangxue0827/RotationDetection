import math
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from ..registry import LOSSES
from .utils import weighted_loss

def smooth_l1_loss(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


def hungarian_loss_quad(inputs, targets):
    quad_inputs  = inputs.reshape(-1, 4, 2)
    quad_targets = targets.reshape(-1, 4, 2)
    losses = torch.stack(
        [smooth_l1_loss(quad_inputs, quad_targets[:, i, :].unsqueeze(1).repeat(1, 4, 1)).sum(2) \
            for i in range(4)] , 1
            )
    indices = [linear_sum_assignment(loss.cpu().detach().numpy()) for loss in losses]
    match_loss = []
    for cnt, (row_ind,col_ind) in enumerate(indices):
        match_loss.append(losses[cnt, row_ind, col_ind])
    return torch.stack(match_loss).sum(1)
#####

@LOSSES.register_module
class HungarianLoss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', form='quad', loss_weight=1.0):
        super(HungarianLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.form = form

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        valid_idx = weight.nonzero()[:,0].unique()
        if len(valid_idx) == 0:
            return torch.tensor(0).float().cuda()
        else:
            if self.form == 'quad':
                loss = hungarian_loss_quad(pred[valid_idx], target[valid_idx].float()) * self.loss_weight
            else:
                raise NotImplementedError
            return loss.sum() / avg_factor
