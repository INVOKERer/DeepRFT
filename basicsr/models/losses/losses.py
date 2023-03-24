import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange
import kornia
from basicsr.models.losses.loss_util import weighted_loss
# from kornia.constants import pi
_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)

rgb_from_hed = torch.tensor([[0.65, 0.70, 0.29],
                            [0.07, 0.99, 0.11],
                            [0.27, 0.57, 0.78]])
# hed_from_rgb = linalg.inv(rgb_from_hed)
hed_from_rgb = torch.linalg.inv(rgb_from_hed)


def separate_stains(rgb, conv_matrix):
    # rgb = _prepare_colorarray(rgb, force_copy=True, channel_axis=-1)
    # rgb = rgb.astype(np.float32)

    # rgb = torch.maximum(rgb, torch.tensor(1e-6))  # avoiding log artifacts
    rgb = torch.clamp(rgb, min=1e-6)
    log_adjust = torch.log(torch.tensor(1e-6))  # used to compensate the sum above
    rgb = rearrange(rgb, 'b c h w -> b h w c')
    stains = (torch.log(rgb) / log_adjust) @ conv_matrix
    stains = rearrange(stains, 'b h w c -> b c h w')
    # stains = torch.maximum(stains, torch.tensor(0.))
    stains = torch.clamp(stains, min=0.)
    return stains

def combine_stains(stains, conv_matrix):

    # stains = stains.astype(np.float32)

    # log_adjust here is used to compensate the sum within separate_stains().
    log_adjust = -torch.log(torch.tensor(1e-6))
    stains = rearrange(stains, 'b c h w -> b h w c')
    log_rgb = -(stains * log_adjust) @ conv_matrix
    rgb = torch.exp(log_rgb)
    rgb = rearrange(rgb, 'b h w c -> b c h w')
    return torch.clamp(rgb, min=0., max=1.)

class rgb2hed(nn.Module):
    def __init__(self,):
        super().__init__()
        self.mat = hed_from_rgb

    def forward(self, x):
        return separate_stains(x, self.mat.to(x.device))
class hed2rgb(nn.Module):
    def __init__(self,):
        super().__init__()
        self.mat = rgb_from_hed

    def forward(self, x):
        return combine_stains(x, self.mat.to(x.device))
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        if isinstance(pred, list):
            loss = 0.
            for predi in pred:
                loss += l1_loss(
                predi, target, weight, reduction=self.reduction)
            return self.loss_weight * loss
        else:
            return self.loss_weight * l1_loss(
                pred, target, weight, reduction=self.reduction)
class L1LossPry(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1LossPry, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        loss = 0.
        target_pry = kornia.geometry.build_pyramid(target, len(pred))
        loss += l1_loss(
            pred[-1], target, weight, reduction=self.reduction)
        pred.pop(-1)
        for i, predi in enumerate(pred):
            loss += l1_loss(
            predi, target_pry[i], weight, reduction=self.reduction) * 0.33

        return self.loss_weight * loss

class FreqLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)

    def forward(self, pred, target):
        diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)
        loss = torch.mean(torch.abs(diff))
        # print(loss)
        return self.loss_weight * loss * 0.01 + self.l1_loss(pred, target)
class CharFreqLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CharFreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = CharbonnierLoss(loss_weight, reduction)

    def forward(self, pred, target):
        diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)
        loss = torch.mean(torch.abs(diff))
        # print(loss)
        return self.loss_weight * loss * 0.01 + self.l1_loss(pred, target)
class StainLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(StainLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)
        self.rgb2hed = rgb2hed()

    def forward(self, pred, target):
        hed = self.rgb2hed(pred)
        Hem, Eos, DAB = torch.chunk(hed, 3, dim=1)
        hedx = self.rgb2hed(target)
        Hemx, Eosx, DABx = torch.chunk(hedx, 3, dim=1)
        HE = torch.cat([Hem, Eos], dim=1)
        HEx = torch.cat([Hemx, Eosx], dim=1)
        diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)
        loss = torch.mean(torch.abs(diff))
        # print(loss)
        return self.loss_weight * loss * 0.01 + self.l1_loss(pred, target) + self.l1_loss(HE, HEx)
class PhasefreqLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(PhasefreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)
    def forward(self, pred, target):
        pred_f = torch.fft.rfft2(pred)
        pred_angle = torch.angle(pred_f)
        tar_f = torch.fft.rfft2(target)
        phase_diff = torch.abs(torch.angle(tar_f) - pred_angle)
        loss = torch.mean(phase_diff)
        return self.loss_weight * loss * 0.01 # + self.l1_loss(pred, target)
class Phase2freqLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(Phase2freqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)
    def forward(self, pred, target):
        pred_f = torch.fft.rfft2(pred)
        tar_f = torch.fft.rfft2(target)
        pred_angle = torch.angle(pred_f)
        tar_angle = torch.angle(tar_f)
        # pred_mag = torch.abs(pred_f)
        p_f = torch.exp(1j * pred_angle) # pred_mag * torch.exp(1j * pred_angle)
        t_f = torch.exp(1j * tar_angle) # pred_mag * torch.exp(1j * tar_angle)
        pred_if = torch.fft.irfft2(p_f)
        target_if = torch.fft.irfft2(t_f)
        # var = torch.var(pred_if, dim=1)
        phase_diff = torch.abs(p_f - t_f)
        # diff = torch.abs(tar_f-pred_f)
        loss = torch.mean(phase_diff)
        return (self.l1_loss(pred_if, target_if) + loss * 0.05) * self.loss_weight# self.loss_weight * loss * 0.01 +
class Phase3freqLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(Phase3freqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        # self.loss_weight = loss_weight
        # self.reduction = reduction
        # self.freq_loss1 = Phase2freqLoss(loss_weight, reduction)
        self.freq_loss2 = Phase2freqLoss(loss_weight, reduction)
    def forward(self, pred, target):
        pred1, pred2 = pred

        return torch.mean(torch.abs(pred1)) + self.freq_loss2(pred2, target)# self.loss_weight * loss * 0.01 +

class MultiFreqLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MultiFreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        # self.loss_weight = loss_weight
        # self.reduction = reduction
        # self.freq_loss1 = Phase2freqLoss(loss_weight, reduction)
        self.freq_loss = FreqLoss(loss_weight, reduction)
    def forward(self, pred, target):
        loss = 0.
        for predi in pred:
            loss += self.freq_loss(predi, target)
        return loss
class MultiScaleFreqLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.freq_loss = FreqLoss(loss_weight, reduction)

    def forward(self, pred, target):
        tar = target
        loss = 0.
        for predi in pred[::-1]:
            loss += self.freq_loss(predi, tar)
            tar = F.interpolate(tar, scale_factor=0.5)
        return loss
class FocalfreqsinLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FocalfreqsinLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)
    def forward(self, pred, target):
        pred_f = torch.fft.rfft2(pred)
        tar_f = torch.fft.rfft2(target)
        diff = pred_f - tar_f
        phase_diff = torch.abs(torch.angle(tar_f) - torch.angle(pred_f))
        # phase_diff = phase_diff / (2 * 3.1415926536)
        phase_diff = torch.sin(phase_diff / 2.)
        loss = torch.mean(torch.abs(diff) * 0.01 + torch.pow(phase_diff, 2) * 0.1)
        return self.loss_weight * loss + self.l1_loss(pred, target)
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
