# --------------------------------------------------------
# gwd pytorch version

# Author: Xue Yang <yangxue-2019-sjtu@sjtu.edu.cn>
#
# License: Apache-2.0 license
# --------------------------------------------------------


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


# from ..builder import LOSSES
# from .utils import weighted_loss


def trace(A):
    return A.diagonal(dim1=-2, dim2=-1).sum(-1)


def sqrt_newton_schulz_autograd(A, numIters, dtype):
    batchSize = A.data.shape[0]
    dim = A.data.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
    I = Variable(torch.eye(dim, dim).view(1, dim, dim).
                 repeat(batchSize, 1, 1).type(dtype), requires_grad=False)
    Z = Variable(torch.eye(dim, dim).view(1, dim, dim).
                 repeat(batchSize, 1, 1).type(dtype), requires_grad=False)

    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA


def wasserstein_distance_sigma(sigma1, sigma2):
    wasserstein_distance_item2 = torch.matmul(sigma1, sigma1) + torch.matmul(sigma2,
                                                                             sigma2) - 2 * sqrt_newton_schulz_autograd(
        torch.matmul(torch.matmul(sigma1, torch.matmul(sigma2, sigma2)), sigma1), 20, torch.FloatTensor)
    wasserstein_distance_item2 = trace(wasserstein_distance_item2)

    return wasserstein_distance_item2


# @weighted_loss
def gwds_loss(pred, target, weight, eps=1e-6):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (Tensor): Predicted bboxes of format (xc, yc, w, h, a),
            shape (n, 5).
        target (Tensor): Corresponding gt bboxes, shape (n, 5).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    mask = (weight > 0).detach()
    pred = pred[mask]
    target = target[mask]

    x1 = pred[:, 0]
    y1 = pred[:, 1]
    w1 = pred[:, 2]
    h1 = pred[:, 3]
    theta1 = pred[:, 4]

    sigma1_1 = w1 / 2 * torch.cos(theta1) ** 2 + h1 / 2 * torch.sin(theta1) ** 2
    sigma1_2 = w1 / 2 * torch.sin(theta1) * torch.cos(theta1) - h1 / 2 * torch.sin(theta1) * torch.cos(theta1)
    sigma1_3 = w1 / 2 * torch.sin(theta1) * torch.cos(theta1) - h1 / 2 * torch.sin(theta1) * torch.cos(theta1)
    sigma1_4 = w1 / 2 * torch.sin(theta1) ** 2 + h1 / 2 * torch.cos(theta1) ** 2
    sigma1 = torch.reshape(
        torch.cat((sigma1_1.unsqueeze(1), sigma1_2.unsqueeze(1), sigma1_3.unsqueeze(1), sigma1_4.unsqueeze(1)), axis=1),
        (-1, 2, 2))

    x2 = target[:, 0]
    y2 = target[:, 1]
    w2 = target[:, 2]
    h2 = target[:, 3]
    theta2 = target[:, 4]
    sigma2_1 = w2 / 2 * torch.cos(theta2) ** 2 + h2 / 2 * torch.sin(theta2) ** 2
    sigma2_2 = w2 / 2 * torch.sin(theta2) * torch.cos(theta2) - h2 / 2 * torch.sin(theta2) * torch.cos(theta2)
    sigma2_3 = w2 / 2 * torch.sin(theta2) * torch.cos(theta2) - h2 / 2 * torch.sin(theta2) * torch.cos(theta2)
    sigma2_4 = w2 / 2 * torch.sin(theta2) ** 2 + h2 / 2 * torch.cos(theta2) ** 2
    sigma2 = torch.reshape(
        torch.cat((sigma2_1.unsqueeze(1), sigma2_2.unsqueeze(1), sigma2_3.unsqueeze(1), sigma2_4.unsqueeze(1)), axis=1),
        (-1, 2, 2))

    wasserstein_distance_item1 = (x1 - x2) ** 2 + (y1 - y2) ** 2
    wasserstein_distance_item2 = wasserstein_distance_sigma(sigma1, sigma2)
    wasserstein_distance = torch.max(wasserstein_distance_item1 + wasserstein_distance_item2,
                                     Variable(torch.zeros(wasserstein_distance_item1.shape[0]).type(torch.FloatTensor),
                                              requires_grad=False))
    wasserstein_distance = torch.max(torch.sqrt(wasserstein_distance + eps),
                                     Variable(torch.zeros(wasserstein_distance_item1.shape[0]).type(torch.FloatTensor),
                                              requires_grad=False))
    wasserstein_similarity = 1 / (wasserstein_distance + 2)
    wasserstein_loss = 1 - wasserstein_similarity

    return wasserstein_loss


def xywhr2xyrs(xywhr):
    xywhr = xywhr.reshape(-1, 5)
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)
    return xy, R, S


def gwd_loss(pred, target, fun='sqrt', tau=1.0, alpha=1.0, normalize=False):
    """
    given any positive-definite symmetrical 2*2 matrix Z:
    Tr(Z^(1/2)) = sqrt(λ_1) + sqrt(λ_2)
    where λ_1 and λ_2 are the eigen values of Z

    meanwhile we have:
    Tr(Z) = λ_1 + λ_2
    det(Z) = λ_1 * λ_2

    combination with following formula:
    (sqrt(λ_1) + sqrt(λ_2))^2 = λ_1 + λ_2 + 2 * sqrt(λ_1 * λ_2)

    yield:
    Tr(Z^(1/2)) = sqrt(Tr(Z) + 2 * sqrt(det(Z)))

    for gwd loss the frustrating coupling part is:
    Tr((Σp^(1/2) * Σt * Σp^(1/2))^(1/2))

    assuming Z = Σp^(1/2) * Σt * Σp^(1/2) then:
    Tr(Z) = Tr(Σp^(1/2) * Σt * Σp^(1/2))
    = Tr(Σp^(1/2) * Σp^(1/2) * Σt)
    = Tr(Σp * Σt)
    det(Z) = det(Σp^(1/2) * Σt * Σp^(1/2))
    = det(Σp^(1/2)) * det(Σt) * det(Σp^(1/2))
    = det(Σp * Σt)

    and thus we can rewrite the coupling part as:
    Tr((Σp^(1/2) * Σt * Σp^(1/2))^(1/2))
    = Tr{Z^(1/2)} = sqrt(Tr(Z) + 2 * sqrt(det(Z))
    = sqrt(Tr(Σp * Σt) + 2 * sqrt(det(Σp * Σt)))
    """
    xy_p, R_p, S_p = xywhr2xyrs(pred)
    xy_t, R_t, S_t = xywhr2xyrs(target)

    xy_distance = (xy_p - xy_t).square().sum(dim=-1)

    Sigma_p = R_p.matmul(S_p.square()).matmul(R_p.permute(0, 2, 1))
    Sigma_t = R_t.matmul(S_t.square()).matmul(R_t.permute(0, 2, 1))

    whr_distance = S_p.diagonal(dim1=-2, dim2=-1).square().sum(dim=-1)
    whr_distance = whr_distance + S_t.diagonal(dim1=-2, dim2=-1).square().sum(
        dim=-1)
    _t = Sigma_p.matmul(Sigma_t)

    _t_tr = _t.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = S_p.diagonal(dim1=-2, dim2=-1).prod(dim=-1)
    _t_det_sqrt = _t_det_sqrt * S_t.diagonal(dim1=-2, dim2=-1).prod(dim=-1)
    whr_distance = whr_distance + (-2) * ((_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt())

    distance = (xy_distance + alpha * alpha * whr_distance).clamp(0)
    # distance = (xy_distance + alpha * alpha * whr_distance).clamp(0).sqrt()

    if normalize:
        wh_p = pred[..., 2:4].clamp(min=1e-7, max=1e7)
        wh_t = target[..., 2:4].clamp(min=1e-7, max=1e7)
        scale = ((wh_p.log() + wh_t.log()).sum(dim=-1) / 4).exp()
        distance = distance / scale

    if fun == 'log':
        distance = torch.log1p(distance)
    elif fun == 'sqrt':
        distance = torch.sqrt(distance)
    else:
        raise ValueError('Invalid non-linear function {fun} for gwd loss')

    if tau >= 1.0:
        return 1 - 1 / (tau + distance)
    else:
        return distance

    # @LOSSES.register_module()
    # class GWDLoss(nn.Module):

    #     def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
    #         super(GWDLoss, self).__init__()
    #         self.eps = eps
    #         self.reduction = reduction
    #         self.loss_weight = loss_weight

    #     def forward(self,
    #                 pred,
    #                 target,
    #                 weight=None,
    #                 avg_factor=None,
    #                 reduction_override=None,
    #                 **kwargs):
    #         assert reduction_override in (None, 'none', 'mean', 'sum')
    #         reduction = (
    #             reduction_override if reduction_override else self.reduction)
    #         if (weight is not None) and (not torch.any(weight > 0)) and (
    #                 reduction != 'none'):
    #             return (pred * weight).sum()  # 0
    #         if weight is not None and weight.dim() > 1:
    #             assert weight.shape == pred.shape
    #             weight = weight.mean(-1)
    #         loss = self.loss_weight * gwd_loss(
    #             pred,
    #             target,
    #             weight,
    #             eps=self.eps,
    #             # reduction=reduction,
    #             # avg_factor=avg_factor,
    #             **kwargs)
    #         return loss

if __name__ == '__main__':
    # pred = torch.FloatTensor([[50, 50, 10, 70, -30 / 180 * np.pi],
    #                           [50, 50, 10, 70, -30 / 180 * np.pi]])
    # target = torch.FloatTensor([[50, 50, 10, 70, -60 / 180 * np.pi],
    #                             [50, 50, 10, 40, -30 / 180 * np.pi]])

    pred = torch.FloatTensor([[50, 50, 10, 70, -30 / 180 * np.pi],
                              [50, 50, 100, 700, -30 / 180 * np.pi]])
    target = torch.FloatTensor([[10, 40, 10, 70, -30 / 180 * np.pi],
                                [10, 40, 100, 700, -40 / 180 * np.pi]])

    weight = torch.FloatTensor([1.0, 1.0])

    print(gwd_loss(pred, target))
