import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import init
import numpy as np
import numbers
from kornia import pi
# from fft_conv import *
import kornia
# from common import *
# import scipy
import os
import math
inf = math.inf
def window_partitions(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows


def window_reverses(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    # B = int(windows.shape[0] / (H * W / window_size / window_size))
    # print('B: ', B)
    # print(H // window_size)
    # print(W // window_size)
    C = windows.shape[1]
    # print('C: ', C)
    x = windows.view(-1, H // window_size, W // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x

def window_partitionx(x, window_size):
    _, _, H, W = x.shape
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_partitions(x[:, :, :h, :w], window_size)
    b_main = x_main.shape[0]
    if h == H and w == W:
        return x_main, [b_main]
    if h != H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_r
        x_dd = x[:, :, -window_size:, -window_size:]
        b_dd = x_dd.shape[0] + b_d
        # batch_list = [b_main, b_r, b_d, b_dd]
        return torch.cat([x_main, x_r, x_d, x_dd], dim=0), [b_main, b_r, b_d, b_dd]
    if h == H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        return torch.cat([x_main, x_r], dim=0), [b_main, b_r]
    if h != H and w == W:
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_main
        return torch.cat([x_main, x_d], dim=0), [b_main, b_d]
def window_reversex(windows, window_size, H, W, batch_list):
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_reverses(windows[:batch_list[0], ...], window_size, h, w)
    B, C, _, _ = x_main.shape
    # print('windows: ', windows.shape)
    # print('batch_list: ', batch_list)
    res = torch.zeros([B, C, H, W],device=windows.device)
    res[:, :, :h, :w] = x_main
    if h == H and w == W:
        return res
    if h != H and w != W and len(batch_list) == 4:
        x_dd = window_reverses(windows[batch_list[2]:, ...], window_size, window_size, window_size)
        res[:, :, h:, w:] = x_dd[:, :, h - H:, w - W:]
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
        x_d = window_reverses(windows[batch_list[1]:batch_list[2], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
        return res
    if w != W and len(batch_list) == 2:
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
    if h != H and len(batch_list) == 2:
        x_d = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
    return res

def window_partitionxy(x, window_size, start=[0, 0]):
    s_h, s_w = start
    assert 0 <= s_h < window_size and 0 <= s_w < window_size
    _, _, H, W = x.shape
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main,  b_main = window_partitionx(x[:, :, s_h:h, s_w:w], window_size)
    if s_h == 0 and s_w == 0:
        return x_main, b_main
    if s_h != 0 and s_w != 0:
        x_l = window_partitions(x[:, :, -h:, :window_size], window_size)
        b_l = x_l.shape[0] + b_main[-1]
        b_main.append(b_l)
        x_u = window_partitions(x[:, :, :window_size, -w:], window_size)
        b_u = x_u.shape[0] + b_l
        b_main.append(b_u)
        x_uu = x[:, :, :window_size, :window_size]
        b_uu = x_uu.shape[0] + b_u
        b_main.append(b_uu)
        # batch_list = [b_main, b_r, b_d, b_dd]
        return torch.cat([x_main, x_l, x_u, x_uu], dim=0), b_main

def window_reversexy(windows, window_size, H, W, batch_list, start=[0, 0]):
    s_h, s_w = start
    assert 0 <= s_h < window_size and 0 <= s_w < window_size

    if s_h == 0 and s_w == 0:
        x_main = window_reversex(windows, window_size, H, W, batch_list)
        return x_main
    else:
        h, w = window_size * (H // window_size), window_size * (W // window_size)
        x_main = window_reversex(windows[:batch_list[-4], ...], window_size, H-s_h, W-s_w, batch_list[:-3])
        B, C, _, _ = x_main.shape
        res = torch.zeros([B, C, H, W], device=windows.device)
        x_uu = window_reverses(windows[batch_list[-2]:, ...], window_size, window_size, window_size)
        res[:, :, :window_size, :window_size] = x_uu[:, :, :, :]
        x_l = window_reverses(windows[batch_list[-4]:batch_list[-3], ...], window_size, h, window_size)
        res[:, :, -h:, :window_size] = x_l
        x_u = window_reverses(windows[batch_list[-3]:batch_list[-2], ...], window_size, window_size, w)
        res[:, :, :window_size, -w:] = x_u[:, :, :, :]

        res[:, :, s_h:, s_w:] = x_main
        return res
def cov(x, mean=True):
    D = x.shape[-1]
    if mean:
        mean = torch.mean(x, dim=-1, keepdim=True)
        x = x - mean
    return 1 / (D - 1) * x @ x.transpose(-1, -2)

def cov_xy(x, y, mean=True):
    D = x.shape[-1]
    if mean:
        mean_x = torch.mean(x, dim=-1, keepdim=True)
        mean_y = torch.mean(x, dim=-1, keepdim=True)
        x = x - mean_x
        y = y - mean_y
    return 1 / (D - 1) * x @ y.transpose(-1, -2)


# def build_filter(pos, freq, POS):
#     result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
#     if freq == 0:
#         return result
#     else:
#         return result * math.sqrt(2)



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        # 维度变换之后必须要使用.contiguous()使得张量在内存连续之后才能调用view函数
        return x.view(int(N/g),g,C,H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        # 维度变换之后必须要使用.contiguous()使得张量在内存连续之后才能调用view函数
        return x.view(N, int(C/g),g,H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)
def get_dctMatrix(m, n):
    N = n
    C_temp = np.zeros([m, n])
    C_temp[0, :] = 1 * np.sqrt(1 / N)

    for i in range(1, m):
        for j in range(n):
            C_temp[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * N)
                                  ) * np.sqrt(2 / N)
    return torch.tensor(C_temp, dtype=torch.float)

def dct2d(feature, dctMat):
    # x = feature.cpu().detach().numpy()
    # out = torch.einsum("...mn,mn->...mn", feature, dctMat)
    # x = scipy.fft.dctn(x, type=2, axes=[-2, -1])
    # out = torch.einsum("...mn,mn->...mn", out, dctMat.T)
    # print(dctMat.shape)
    feature = dctMat @ feature
    feature = feature @ dctMat.T
    return feature # torch.tensor(x, device=feature.device)

def idct2d(feature, dctMat):
    # x = feature.cpu().detach().numpy()
    # x = scipy.fft.idctn(x, type=2, axes=[-2, -1])
    # out = torch.einsum("...mn,mn->...mn", feature, dctMat)
    # out = torch.einsum("...mn,mn->...mn", out, dctMat.T)
    feature = dctMat.T @ feature
    feature = feature @ dctMat
    return feature # torch.tensor(x, device=feature.device)
def get_Coord(ins_feat):
    x_range = torch.linspace(-1, 1, ins_feat.shape[-1], device=ins_feat.device)
    y_range = torch.linspace(-1, 1, ins_feat.shape[-2], device=ins_feat.device)
    y, x = torch.meshgrid(y_range, x_range)
    y = y.expand([ins_feat.shape[0], 1, -1, -1])
    x = x.expand([ins_feat.shape[0], 1, -1, -1])
    coord_feat = torch.cat([x, y], 1)
    ins_feat = torch.cat([ins_feat, coord_feat], 1)
    return ins_feat
class DCT2(nn.Module):
    def __init__(self, window=256):
        super(DCT2, self).__init__()
        self.dctMat = get_dctMatrix(window, window)
    def forward(self, x):
        self.dctMat = self.dctMat.to(x.device)
        # print(x.shape, self.dctMat.shape)
        x = dct2d(x, self.dctMat)
        return x
class IDCT2(nn.Module):
    def __init__(self, window=256):
        super(IDCT2, self).__init__()
        self.dctMat = get_dctMatrix(window, window)
    def forward(self, x):
        self.dctMat = self.dctMat.to(x.device)
        x = idct2d(x, self.dctMat)
        return x
class DCT_test(nn.Module):
    def __init__(self, window):
        super(DCT_test, self).__init__()
        self.dctMat = get_dctMatrix(window, window)
        # self.dctMat = self.dctMat.to(self.device)
        # print(self.dctMat)
        # self.dctMat.cuda()
    def forward(self, x):
        # print(self.dctMat.shape)
        # _, _, m, n = x.shape
        self.dctMat = self.dctMat.to(x.device)
        # y = x.cpu().numpy()
        x_dct = dct2d(x, self.dctMat)
        x_idct = idct2d(x_dct, self.dctMat)
        # x_idct = torch.tensor(x_idct, device=x.device)
        return x_idct, x_dct


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, relu_method=nn.ReLU, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))

        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class BasicConv_do(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bias=False, norm=False, relu=True, transpose=False,
                 relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d):
        super(BasicConv_do, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                DOConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class BasicConv_do_eval(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d):
        super(BasicConv_do_eval, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                DOConv2d_eval(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ResBlock(nn.Module):
    def __init__(self, out_channel, kernel_size=3):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=kernel_size, stride=1, relu=True, norm=False),
            BasicConv(out_channel, out_channel, kernel_size=kernel_size, stride=1, relu=False, norm=False)
        )

    def forward(self, x):
        y = self.main(x)
        return y + x






##########################################################################
# class TransformerBlock(nn.Module):
#     def __init__(self, out_channel, num_heads=1, bias=False, norm='backward'):
#         super(TransformerBlock, self).__init__()
#
#         # self.norm1 = LayerNorm(dim, LayerNorm_type)
#         self.attn = Attention(out_channel, num_heads, bias, norm)
#         # self.norm2 = LayerNorm(dim, LayerNorm_type)
#         # self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
#         self.res_dwconv = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, groups=1, bias=bias),
#             nn.ReLU(),
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, groups=1, bias=bias)
#         )
#         self.conv = nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1, bias=bias)
#         # self.norm = norm
#
#     def forward(self, x):
#
#         y = self.attn(x)
#
#         y = self.conv(y)
#         x = x + y + self.res_dwconv(x)
#         # x = x + self.ffn(x)
#         return x

class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, conv=BasicConv):
        super().__init__()
        self.conv1 = nn.Sequential(
            conv(dim, hidden_dim, kernel_size=1, stride=1, relu=False, bias=False),
            act_layer()
        )
        self.dwconv = nn.Sequential(
            conv(hidden_dim, hidden_dim, kernel_size=3, stride=1, groups=hidden_dim, relu=False, bias=False),
            act_layer()
        )
        self.conv2 = nn.Sequential(
            conv(hidden_dim, dim, kernel_size=1, stride=1, relu=False, bias=False)
        )
        # self.dim = dim
        # self.hidden_dim = hidden_dim

    def forward(self, x):
        # bs x hw x c
        x = self.conv1(x)
        # bs,hidden_dim,32x32
        x = self.dwconv(x)
        x = self.conv2(x)
        return x



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Trans_like_RFT(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(Trans_like_RFT, self).__init__()
        self.attn = ResBlock_fft_window_bench(out_channel, win=32, norm=norm)
        self.ffn = FeedForward(out_channel)

    def forward(self, x):

        return self.ffn(self.attn(x))


class RTGB(nn.Module):
    def __init__(self, out_channel, h, w):
        super(RTGB, self).__init__()
        height = h
        width = w
        self.channel_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
            # nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=False)
        )
        self.wide_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(height, height, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
            # nn.Conv2d(height, height, kernel_size=1, stride=1, bias=False)
        )

        self.high_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(width, width, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
            # nn.Conv2d(width, width, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        c = self.channel_gap(x)
        h = self.high_gap(x.transpose(-3, -2))
        w = self.wide_gap(x.transpose(-3, -1))
        ch = torch.einsum("...cbd,...hbd->...chbd", c, h)
        # print(ch)
        chw = torch.einsum("...chbd,...wbd->...chw", ch, w)
        # print(chw)
        return chw

class DRTLM(nn.Module):
    def __init__(self, channel, h, w):
        super(DRTLM, self).__init__()
        self.RT1 = RTGB(channel, h, w)
        self.RT2 = RTGB(channel, h, w)
        self.RT3 = RTGB(channel, h, w)
    def forward(self, x):
        O1 = self.RT1(x)
        O2_in = x - O1
        O2 = self.RT2(O2_in)
        O3_in = O1 - O2
        O3 = self.RT3(O3_in)
        O2 = O1 + O2
        O3 + O2 + O3
        return torch.cat([O1, O2, O3], dim=1)

class Res_TLPLN(nn.Module):
    def __init__(self, n_feature=32, h=256, w=256, bias=False):
        super(Res_TLPLN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=n_feature, out_channels=n_feature, kernel_size=3, bias=bias, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_feature, out_channels=n_feature, kernel_size=3, bias=bias, padding=1)
        )
        self.DRT = DRTLM(n_feature, h, w)
        self.conv2 = nn.Conv2d(in_channels=n_feature*3, out_channels=n_feature, kernel_size=3, bias=bias,
                                   padding=1)


    def forward(self, x):
        feat1 = self.conv1(x)
        feat2 = self.conv2(self.DRT(feat1))
        f_out = feat2 * feat1 + x
        # f_out = torch.softmax(feat2, dim=1) * feat1 + x
        return f_out
class Res_TLPLN_win(nn.Module):
    def __init__(self, n_feature=32, window=32, bias=False):
        super(Res_TLPLN_win, self).__init__()
        self.window = window
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=n_feature, out_channels=n_feature, kernel_size=3, bias=bias, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_feature, out_channels=n_feature, kernel_size=3, bias=bias, padding=1)
        )
        self.DRT = DRTLM(n_feature, self.window, self.window)
        self.conv2 = nn.Conv2d(in_channels=n_feature*3, out_channels=n_feature, kernel_size=3, bias=bias,
                                   padding=1)

    def forward(self, x):
        _, _, H, W = x.shape
        feat1 = self.conv1(x)
        featx, batch_list = window_partitionx(feat1, self.window)
        featx = self.DRT(featx)
        featx = window_reversex(featx, self.window, H, W, batch_list)
        feat2 = self.conv2(featx)
        f_out = feat2 * feat1 + x
        # f_out = torch.softmax(feat2, dim=1) * feat1 + x
        return f_out

class Res_TLPLN_win_norm(nn.Module):
    def __init__(self, n_feature=32, h=256, w=256, bias=False, reduce=8):
        super(Res_TLPLN_win_norm, self).__init__()
        self.window = h//reduce
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=n_feature, out_channels=n_feature, kernel_size=3, bias=bias, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_feature, out_channels=n_feature, kernel_size=3, bias=bias, padding=1),
            simam_module()
        )
        self.DRT = DRTLM(n_feature, self.window, self.window)
        self.conv2 = nn.Conv2d(in_channels=n_feature*3, out_channels=n_feature, kernel_size=3, bias=bias,
                                   padding=1)

    def forward(self, x):
        _, _, H, W = x.shape
        feat1 = self.conv1(x)
        featx, batch_list = window_partitionx(feat1, self.window)
        featx = self.DRT(featx)
        featx = window_reversex(featx, self.window, H, W, batch_list)
        feat2 = self.conv2(featx)
        f_out = feat2 * feat1 + x
        # f_out = torch.softmax(feat2, dim=1) * feat1 + x
        return f_out
class ResBlock_only_fft_window_bench(nn.Module):
    def __init__(self, out_channel, win=None, norm='backward'):
        super(ResBlock_only_fft_window_bench, self).__init__()
        self.win = win
        self.norm = norm

        self.main_fft = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel*2, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(out_channel*2, out_channel*2, kernel_size=1, stride=1)
        )
    def forward(self, x):
        _, _, H, W = x.shape
        # local
        if self.win is not None:
            x_windows, batch_list = window_partitionx(x, self.win)
        else:
            x_windows = x
        dim = 1
        y = torch.fft.rfft2(x_windows, norm=self.norm)

        y_f = torch.cat([y.real, y.imag], dim=dim)
        y_f = self.main_fft(y_f)  # + y_f
        y_real, y_imag = torch.chunk(y_f, 2, dim=dim)
        y_f = torch.complex(y_real, y_imag)
        y_f = torch.fft.irfft2(y_f, norm=self.norm)
        if self.win is not None:
            y_f = window_reversex(y_f, self.win, H, W, batch_list)
        return y_f + x

class ResBlock_fft_window_bench(nn.Module):
    def __init__(self, out_channel, win=256, norm='backward'):
        super(ResBlock_fft_window_bench, self).__init__()
        self.win = win
        self.norm = norm

        self.main = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        # self.norm = RepresentativeBatchNorm2d(in_channel)
        self.main_fft = nn.Sequential(
            BasicConv(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=False)

        )
    def forward(self, x):
        _, _, H, W = x.shape
        # local
        x_windows, batch_list = window_partitionx(x, self.win)
        dim = 1
        y = torch.fft.rfft2(x_windows, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)  # + y_f
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, norm=self.norm)
        # y = y + self.main(x_windows)
        y_re = window_reversex(y, self.win, H, W, batch_list)
        return y_re + x + self.main(x)

class ResBlock_dct_bench(nn.Module):
    def __init__(self, out_channel, win=256, norm='backward'):
        super(ResBlock_dct_bench, self).__init__()
        self.win = win
        self.norm = norm

        self.main = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        # self.norm = RepresentativeBatchNorm2d(in_channel)
        self.main_dct = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=1, stride=1, relu=False)
        )
        self.dct = DCT2(window=win)
        self.idct = IDCT2(window=win)
    def forward(self, x):
        _, _, H, W = x.shape
        y = self.dct(x)
        # print(y.min(), y.max())
        y = self.main_dct(y)
        y = self.idct(y)
        return y + x + self.main(x)
class ResBlock_dct_window_bench(nn.Module):
    def __init__(self, out_channel, win=256, norm='backward'):
        super(ResBlock_dct_window_bench, self).__init__()
        self.win = win
        self.norm = norm

        self.main = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        # self.norm = RepresentativeBatchNorm2d(in_channel)
        self.main_dct = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=1, stride=1, relu=False)
        )
        self.dct = DCT2(window=win)
        self.idct = IDCT2(window=win)
    def forward(self, x):
        _, _, H, W = x.shape
        # local
        if H != self.win or W != self.win:
            x_windows, batch_list = window_partitionx(x, self.win)
        else:
            x_windows = x
        y = self.dct(x_windows)
        # print(y.min(), y.max())
        y = self.main_dct(y)
        y = self.idct(y)
        if H != self.win or W != self.win:
            y = window_reversex(y, self.win, H, W, batch_list)
        return y + x + self.main(x)
class dct_window_bench(nn.Module):
    def __init__(self, out_channel, win=256, norm='backward'):
        super(dct_window_bench, self).__init__()
        self.win = win
        self.norm = norm
        # self.norm = RepresentativeBatchNorm2d(in_channel)
        self.main_dct = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=1, stride=1, relu=False)
        )
        self.dct = DCT2(window=win)
        self.idct = IDCT2(window=win)
    def forward(self, x, mask=None):
        _, _, H, W = x.shape
        # local
        if H != self.win or W != self.win:
            x, batch_list = window_partitionx(x, self.win)

        x = self.dct(x)
        if mask is not None:
            x = x * mask
        # print(y.min(), y.max())
        x = self.main_dct(x)
        x = self.idct(x)
        if H != self.win or W != self.win:
            x = window_reversex(x, self.win, H, W, batch_list)
        return x
class ResBlock_do_fft_window_bench(nn.Module):
    def __init__(self, out_channel, win=32, norm='backward'):
        super(ResBlock_do_fft_window_bench, self).__init__()
        self.win = win
        self.norm = norm

        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        # self.norm = RepresentativeBatchNorm2d(in_channel)
        self.main_fft = nn.Sequential(
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=False)

        )
    def forward(self, x):
        _, _, H, W = x.shape
        # local
        x_windows, batch_list = window_partitionx(x, self.win)
        dim = 1
        y = torch.fft.rfft2(x_windows, norm=self.norm)
        # y_imag = y.imag
        # y_real = y.real
        y_f = torch.cat([y.real, y.imag], dim=dim)
        y = self.main_fft(y_f)  # + y_f
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, norm=self.norm)
        # y = y + self.main(x_windows)
        y_re = window_reversex(y, self.win, H, W, batch_list)
        return y_re + x + self.main(x)
class ResBlock_do_fft_window_bench_mv2(nn.Module):
    def __init__(self, out_channel, win=32, norm='backward'):
        super(ResBlock_do_fft_window_bench_mv2, self).__init__()
        self.win = win
        self.norm = norm

        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        # self.norm = RepresentativeBatchNorm2d(in_channel)
        self.main_fft1 = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=1, stride=1, relu=False)
        )
        self.main_fft2 = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=1, stride=1, relu=False)
        )
    def forward(self, x):
        _, _, H, W = x.shape
        x1, x2 = torch.chunk(x, 2, dim=1)
        # local
        x1_windows, batch_list = window_partitionx(x1, self.win)
        start_xy = self.win//2-1
        x2_windows, batch_list2 = window_partitionxy(x1, self.win, [start_xy, start_xy])
        # print(batch_list2)
        dim = 1
        y1 = torch.fft.rfft2(x1_windows, norm=self.norm)
        y1 = torch.cat([y1.real, y1.imag], dim=dim)
        y1 = self.main_fft1(y1)  # + y_f
        y_real, y_imag = torch.chunk(y1, 2, dim=dim)
        y1 = torch.complex(y_real, y_imag)
        y1 = torch.fft.irfft2(y1, norm=self.norm)
        # y = y + self.main(x_windows)
        y1 = window_reversex(y1, self.win, H, W, batch_list)

        y2 = torch.fft.rfft2(x2_windows, norm=self.norm)
        y2 = torch.cat([y2.real, y2.imag], dim=dim)
        y2 = self.main_fft2(y2)  # + y_f
        y_real, y_imag = torch.chunk(y2, 2, dim=dim)
        y2 = torch.complex(y_real, y_imag)
        y2 = torch.fft.irfft2(y2, norm=self.norm)
        # y = y + self.main(x_windows)
        y2 = window_reversexy(y2, self.win, H, W, batch_list2, [start_xy, start_xy])
        y = torch.cat([y1, y2], dim=1)
        return y + x + self.main(x)
class ResBlock_do_fft_window_bench_xy(nn.Module):
    def __init__(self, out_channel, win=32, norm='backward'):
        super(ResBlock_do_fft_window_bench_xy, self).__init__()
        self.win = win
        self.norm = norm
        self.start_xy = win // 2 - 1

        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        # self.norm = RepresentativeBatchNorm2d(in_channel)
        self.main_fft = nn.Sequential(
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=False)
        )

    def forward(self, x):
        _, _, H, W = x.shape
        # x1, x2 = torch.chunk(x, 2, dim=1)
        # local
        x_windows, batch_list = window_partitionxy(x, self.win, [self.start_xy, self.start_xy])
        # print(batch_list2)
        dim = 1
        y2 = torch.fft.rfft2(x_windows, norm=self.norm)
        y2 = torch.cat([y2.real, y2.imag], dim=dim)
        y2 = self.main_fft(y2)  # + y_f
        y_real, y_imag = torch.chunk(y2, 2, dim=dim)
        y2 = torch.complex(y_real, y_imag)
        y2 = torch.fft.irfft2(y2, norm=self.norm)
        # y = y + self.main(x_windows)
        y2 = window_reversexy(y2, self.win, H, W, batch_list, [self.start_xy, self.start_xy])
        return y2 + x + self.main(x)
class ResBlock_do_fft_dct_window_bench(nn.Module):
    def __init__(self, out_channel, win=32, norm='backward'):
        super(ResBlock_do_fft_dct_window_bench, self).__init__()
        self.win = win
        self.norm = norm
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.cat_conv = BasicConv_do(out_channel*2, out_channel, kernel_size=1, stride=1, relu=False)
        # self.norm = RepresentativeBatchNorm2d(in_channel)
        self.main_fft = fft_bench(out_channel)
        self.main_dct = dct_window_bench(out_channel, win)
    def forward(self, x):
        _, _, H, W = x.shape
        x_windows, batch_list = window_partitionx(x, self.win)
        x_windows = torch.cat([self.main_fft(x_windows), self.main_dct(x_windows)], dim=1)
        x_windows = self.cat_conv(x_windows)
        x_windows = window_reversex(x_windows, self.win, H, W, batch_list)
        return x_windows + x + self.main(x)
class ResBlock_do_fft_dct_cat_window_bench(nn.Module):
    def __init__(self, out_channel, win=32, norm='backward'):
        super(ResBlock_do_fft_dct_cat_window_bench, self).__init__()
        self.win = win
        self.norm = norm
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        # self.norm = RepresentativeBatchNorm2d(in_channel)
        self.main_fft = fft_bench(out_channel)
        self.main_dct = dct_window_bench(out_channel, win)
    def forward(self, x):
        _, _, H, W = x.shape
        x_windows, batch_list = window_partitionx(x, self.win)
        x_windows = self.main_fft(x_windows) + self.main_dct(x_windows)
        x_windows = window_reversex(x_windows, self.win, H, W, batch_list)
        return x_windows + x + self.main(x)

class ResBlock_dct_diff_window_bench(nn.Module):
    def __init__(self, out_channel, win=[16, 32], norm='backward'):
        super(ResBlock_dct_diff_window_bench, self).__init__()
        self.win = win
        self.norm = norm
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        # self.norm = RepresentativeBatchNorm2d(in_channel)
        self.main_dct0 = dct_window_bench(out_channel, win[0])
        self.main_dct1 = dct_window_bench(out_channel, win[1])
        
    def forward(self, x):
        x_windows = self.main_dct0(x) + self.main_dct1(x)
        return x_windows + x + self.main(x)

class ResBlock_dct_window_bench(nn.Module):
    def __init__(self, out_channel, win=32, norm='backward'):
        super(ResBlock_dct_window_bench, self).__init__()
        self.win = win
        self.norm = norm
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        # self.norm = RepresentativeBatchNorm2d(in_channel)
        self.main_dct0 = dct_window_bench(out_channel, win)
    def forward(self, x):
        x_windows = self.main_dct0(x)
        return x_windows + x + self.main(x)
class ResBlock_dct_mask_window_bench(nn.Module):
    def __init__(self, out_channel, win=32, norm='backward'):
        super(ResBlock_dct_mask_window_bench, self).__init__()
        self.win = win
        self.norm = norm
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        # self.norm = RepresentativeBatchNorm2d(in_channel)
        self.main_dct0 = dct_window_bench(out_channel, win)
        self.modulater = nn.Parameter(torch.Tensor(out_channel, win, win))
        init.kaiming_uniform_(self.modulater, a=math.sqrt(4))
    def forward(self, x):
        mask = torch.sigmoid(self.modulater)
        x_windows = self.main_dct0(x, mask)
        return x_windows + x + self.main(x)
class ResBlock_do_fft_dct_window_bench_shift(nn.Module):
    def __init__(self, out_channel, dct=True, win=32, norm='backward'):
        super(ResBlock_do_fft_dct_window_bench_shift, self).__init__()
        self.win = win
        self.norm = norm
        self.shift_size = win // 2
        self.dct = dct
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        # self.norm = RepresentativeBatchNorm2d(in_channel)
        self.main_fft = fft_bench(out_channel)
        self.main_dct = dct_window_bench(out_channel, win)
    def forward(self, x):
        _, _, H, W = x.shape
        # x1, x2 = torch.chunk(x, 2, dim=1)
        # local
        x_ = F.pad(x, (self.shift_size, 0, self.shift_size, 0), mode='constant')

        x_windows_1, batch_list_1 = window_partitionx(x_, self.win)
        x_windows_2, batch_list_2 = window_partitionx(x, self.win)
        if self.dct:
            x_windows_1, x_windows_2 = x_windows_2, x_windows_1
        x_windows_1 = self.main_fft(x_windows_1)
        x_windows_2 = self.main_dct(x_windows_2)
        if self.dct:
            x_windows_1, x_windows_2 = x_windows_2, x_windows_1
        x_windows_1 = window_reversex(x_windows_1, self.win, H+self.shift_size, W+self.shift_size, batch_list_1)
        x_windows_2 = window_reversex(x_windows_2, self.win, H, W, batch_list_2)
        # print(x_.shape, x_windows_1.shape, x_windows_2.shape)
        x_windows_1 = x_windows_1[:, :, self.shift_size:, self.shift_size:]
        return x_windows_1 + x_windows_2 + x + self.main(x)
class ResBlock_do_fft_dct_window_bench_xy(nn.Module):
    def __init__(self, out_channel, dct=True, win=32, norm='backward'):
        super(ResBlock_do_fft_dct_window_bench_xy, self).__init__()
        self.win = win
        self.norm = norm
        self.start_xy = win // 2 - 1
        self.dct = dct
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        # self.norm = RepresentativeBatchNorm2d(in_channel)
        self.main_fft = fft_bench(out_channel)
        self.main_dct = dct_window_bench(out_channel, win)
    def forward(self, x):
        _, _, H, W = x.shape
        # x1, x2 = torch.chunk(x, 2, dim=1)
        # local
        x_windows_1, batch_list_1 = window_partitionxy(x, self.win, [self.start_xy, self.start_xy])
        x_windows_2, batch_list_2 = window_partitionx(x, self.win)
        if self.dct:
            x_windows_1, x_windows_2 = x_windows_2, x_windows_1
        x_windows_1 = self.main_fft(x_windows_1)
        x_windows_2 = self.main_dct(x_windows_2)
        if self.dct:
            x_windows_1, x_windows_2 = x_windows_2, x_windows_1
        x_windows_1 = window_reversexy(x_windows_1, self.win, H, W, batch_list_1, [self.start_xy, self.start_xy])
        x_windows_2 = window_reversex(x_windows_2, self.win, H, W, batch_list_2)
        return x_windows_1 + x_windows_2 + x + self.main(x)
class ResBlock_do_dct_fft_window_bench_shift_res(nn.Module):
    def __init__(self, out_channel, win=32, norm='backward'):
        super(ResBlock_do_dct_fft_window_bench_shift_res, self).__init__()
        self.resb1 = ResBlock_do_fft_dct_window_bench_shift(out_channel, dct=True, win=win, norm=norm)
        self.resb2 = ResBlock_do_fft_dct_window_bench_shift(out_channel, dct=False, win=win, norm=norm)

    def forward(self, x):
        x = self.resb1(x)
        x = self.resb2(x)
        return x
class ResBlock_do_dct_fft_window_bench_2_res(nn.Module):
    def __init__(self, out_channel, win=32, norm='backward'):
        super(ResBlock_do_dct_fft_window_bench_2_res, self).__init__()
        self.resb1 = ResBlock_do_fft_dct_window_bench_xy(out_channel, dct=True, win=win, norm=norm)
        self.resb2 = ResBlock_do_fft_dct_window_bench_xy(out_channel, dct=False, win=win, norm=norm)

    def forward(self, x):
        x = self.resb1(x)
        x = self.resb2(x)
        return x
class ResBlock_do_fft_window_bench_mv2_cll(nn.Module):
    def __init__(self, out_channel, win=32, norm='backward'):
        super(ResBlock_do_fft_window_bench_mv2_cll, self).__init__()
        self.resb1 = ResBlock_do_fft_window_bench(out_channel, win, norm)
        self.resb2 = ResBlock_do_fft_window_bench_xy(out_channel, win, norm)

    def forward(self, x):
        x = self.resb1(x)
        x = self.resb2(x)
        return x
class ResBlock_do_fft_window_bench_modulater(nn.Module):
    def __init__(self, out_channel, win=32, norm='backward'):
        super(ResBlock_do_fft_window_bench_modulater, self).__init__()
        self.win = win
        self.norm = norm

        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        # self.norm = RepresentativeBatchNorm2d(in_channel)
        self.main_fft = nn.Sequential(
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=False)
        )
        self.modulater = nn.Parameter(torch.randn(out_channel, win, win//2+1, 2, dtype=torch.float32) * 0.02)
        # self.modulater = nn.Parameter(torch.randn(out_channel, win, win//2+1, dtype=torch.float32) * 0.02)
    def forward(self, x):
        _, _, H, W = x.shape
        # local
        x_windows, batch_list = window_partitionx(x, self.win)
        dim = 1
        y = torch.fft.rfft2(x_windows, norm=self.norm)
        modulater = torch.view_as_complex(self.modulater)
        # modu = torch.sigmoid(self.modulater)
        y = y + modulater
        y_f = torch.cat([y.real, y.imag], dim=dim)
        y = self.main_fft(y_f)  # + y_f
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y_real = y_real
        y_imag = y_imag
        y = torch.complex(y_real, y_imag)

        # print(modulater.shape, y.shape)

        # y = y + modu
        y = torch.fft.irfft2(y, norm=self.norm)
        # y = y + self.main(x_windows)
        y_re = window_reversex(y, self.win, H, W, batch_list)
        return y_re + x + self.main(x)
class ResBlock_do_fft_window_bench_r(nn.Module):
    def __init__(self, out_channel, win=32, norm='backward'):
        super(ResBlock_do_fft_window_bench_r, self).__init__()
        self.win = win
        self.norm = norm

        # self.main = nn.Sequential(
        #     BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
        #     BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        # )
        self.norm1 = nn.LayerNorm(out_channel)
        self.norm2 = nn.LayerNorm(out_channel)
        self.relu = nn.ReLU()
        self.main_fft = nn.Sequential(
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=False)
        )
        h_dim = int(out_channel)
        self.ffn = LeFF(dim=out_channel, hidden_dim=h_dim, conv=BasicConv_do,act_layer=nn.ReLU)
    def forward(self, x):
        _, _, H, W = x.shape
        x_ln = rearrange(x, 'b c h w -> b (h w) c')
        x_ln = self.norm1(x_ln)
        x_ln = rearrange(x_ln, 'b (h w) c -> b c h w',h=H,w=W)
        # local
        x_windows, batch_list = window_partitionx(x_ln, self.win)
        dim = 1
        y = torch.fft.rfft2(x_windows, norm=self.norm)
        # y_imag = y.imag
        # y_real = y.real
        y_f = torch.cat([y.real, y.imag], dim=dim)
        y = self.main_fft(y_f)  # + y_f
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, norm=self.norm)
        # y = y + self.main(x_windows)
        y_re = window_reversex(y, self.win, H, W, batch_list)
        y_re = self.relu(y_re) + x
        y_re = rearrange(y_re, 'b c h w -> b (h w) c')
        y_re = self.norm1(y_re)
        y_re = rearrange(y_re, 'b (h w) c -> b c h w', h=H, w=W)
        y_re = self.ffn(y_re) + y_re
        return y_re  # + self.main(x)
class ResBlock_do_fft_window_mlp_bench(nn.Module):
    def __init__(self, out_channel, win=32, norm='backward'):
        super(ResBlock_do_fft_window_mlp_bench, self).__init__()
        self.win = win
        self.norm = norm
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.complex_weight1 = nn.Parameter(torch.randn(out_channel, win, win//2+1, 2, dtype=torch.float32) * 0.02)
        # self.complex_weight2 = nn.Parameter(torch.randn(out_channel, win, win // 2 + 1, 2, dtype=torch.float32) * 0.02)
        self.act = nn.ReLU()
        self.main_fft = nn.Sequential(
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=False)

        )
    def forward(self, x):
        weight1 = torch.view_as_complex(self.complex_weight1)
        # weight2 = torch.view_as_complex(self.complex_weight2)
        _, _, H, W = x.shape
        # local
        x_windows, batch_list = window_partitionx(x, self.win)
        dim = 1
        y = torch.fft.rfft2(x_windows, norm=self.norm)
        # print(y.shape, weight.shape)
        x_ = y * weight1

        y_f = torch.cat([y.real, y.imag], dim=dim)
        y = self.main_fft(y_f)  # + y_f
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y + x_, norm=self.norm)
        # y = y + self.main(x_windows)
        y_re = window_reversex(y, self.win, H, W, batch_list)
        return y_re + x + self.main(x)


class ResBlock_do_fft_window_cyclic_bench(nn.Module):
    def __init__(self, out_channel, win=256, shift_size=256//2, norm='backward'):
        super(ResBlock_do_fft_window_cyclic_bench, self).__init__()
        self.win = win
        self.norm = norm

        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.shift_size = shift_size
        # self.norm = RepresentativeBatchNorm2d(in_channel)
        self.main_fft = nn.Sequential(
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=False)

        )
    def forward(self, x):
        _, _, H, W = x.shape
        # local
        # cyclic shift
        if self.shift_size != 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(-1, -2))
        else:
            shifted_x = x
        x_windows, batch_list = window_partitionx(shifted_x, self.win)
        dim = 1
        y = torch.fft.rfft2(x_windows, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)  # + y_f
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, norm=self.norm)
        # y = y + self.main(x_windows)
        y_re = window_reversex(y, self.win, H, W, batch_list)
        if self.shift_size > 0:
            y_re = torch.roll(y_re, shifts=(self.shift_size, self.shift_size), dims=(-1, -2))
        return y_re + x + self.main(x)

class ResBlock_do_fft_window_pad_bench(nn.Module):
    def __init__(self, out_channel, win=256, shift_size=256//2, norm='backward'):
        super(ResBlock_do_fft_window_pad_bench, self).__init__()
        self.win = win
        self.norm = norm

        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.shift_size = shift_size
        # self.norm = RepresentativeBatchNorm2d(in_channel)
        self.main_fft = nn.Sequential(
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=False)

        )
    def forward(self, x):
        _, _, H, W = x.shape
        # local
        # cyclic shift
        if self.shift_size != 0:
            shifted_x = F.pad(x, (self.shift_size, 0, self.shift_size, 0), mode='constant')#, mode='constant')
        else:
            shifted_x = x
        x_windows, batch_list = window_partitionx(shifted_x, self.win)
        # print(x_windows.shape)
        dim = 1
        y = torch.fft.rfft2(x_windows, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)  # + y_f
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, norm=self.norm)
        # y = y + self.main(x_windows)
        # print(y.shape)

        if self.shift_size > 0:
            y_re = window_reversex(y, self.win, H+self.shift_size, W+self.shift_size, batch_list)
            y_re = y_re[:, :, self.shift_size:, self.shift_size:]
        else:
            y_re = window_reversex(y, self.win, H, W, batch_list)
        return y_re + x + self.main(x)
class ResBlock_do_fft_window_bench_withcyclic(nn.Module):
    def __init__(self, out_channel, win=256, norm='backward'):
        super(ResBlock_do_fft_window_bench_withcyclic, self).__init__()
        self.win = win
        self.main = nn.Sequential(
            ResBlock_do_fft_window_cyclic_bench(out_channel, win=win, shift_size=0, norm=norm),
            ResBlock_do_fft_window_cyclic_bench(out_channel, win=win, shift_size=win//2, norm=norm)
        )

    def forward(self, x):
        return self.main(x)
class ResBlock_fft_window_benchx(nn.Module):
    def __init__(self, in_channel, out_channel, win=8, norm='backward'):
        super(ResBlock_fft_window_benchx, self).__init__()
        self.win = win
        self.norm = norm

        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=False)
        )
        # self.norm = RepresentativeBatchNorm2d(in_channel)
        self.main_fft = nn.Sequential(
            BasicConv(in_channel*2, out_channel*2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=False)

        )
    def forward(self, x):
        _, _, H, W = x.shape
        # local
        x_windows, batch_list = window_partitionx(x, self.win)
        dim = 1
        y = torch.fft.rfft2(x_windows, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)  # + y_f
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, norm=self.norm)
        # y = y + self.main(x_windows)
        y_re = window_reversex(y, self.win, H, W, batch_list)
        return y_re + self.conv(x) + self.main(x)

class ResBlock_do(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock_do, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x

class ResBlock_do_eval(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock_do_eval, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x

class ResBlock_fft_channel_bench(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_fft_channel_bench, self).__init__()
        dim = (out_channel // 2 + 1)
        self.realconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, stride=1, bias=False)
        )
        self.imagconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, stride=1, bias=False)
        )
        self.norm = norm
    def forward(self, x):
        _, C, _, _ = x.shape
        # dim = 1
        y = torch.fft.rfft(x, dim=1, norm=self.norm)
        # y_f = torch.cat([y.real, y.imag], dim=dim)
        # print(y_f.shape, y_f.dtype)
        y.real = self.realconv(y.real)
        y.imag = self.imagconv(y.imag)
        # y_real, y_imag = torch.chunk(y_f, 2, dim=dim)
        # y = torch.complex(y_real, y_imag)
        # print(y.shape)
        y = torch.fft.irfft(y, dim=1, norm=self.norm)
        # print(y.shape)
        return x + y

class ResBlock_fft_channel_bench_ED2(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_fft_channel_bench_ED2, self).__init__()
        dim = (out_channel // 2 + 1)
        self.main1 = complex_conv(dim, relu=True)
        self.main2 = complex_conv(dim)

    def forward(self, x):

        y_real, y_imag = self.main1(x.real, x.imag)
        y_real, y_imag = self.main2(y_real, y_imag)
        y = torch.complex(y_real, y_imag)
        return x + y
class complex_conv(nn.Module):
    def __init__(self, dim, relu=False, norm='backward'):
        super(complex_conv, self).__init__()
        self.realconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, stride=1, bias=False),
        )
        self.imagconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, stride=1, bias=False),
        )
        self.relu = relu
    def forward(self, x_real, x_imag):

        y_real = self.realconv(x_real) - self.imagconv(x_imag)
        y_imag = self.imagconv(x_real) + self.realconv(x_imag)
        if self.relu:
            y_real = torch.relu(y_real)
            y_imag = torch.relu(y_imag)
        return y_real, y_imag
class ResBlock_do_fft_bench(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_do_fft_bench, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft = nn.Sequential(
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=False)
        )
        self.dim = out_channel
        self.norm = norm
    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        # y_imag = y.imag
        # y_real = y.real
        y_f = torch.cat([y.real, y.imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y

class ResBlock_1x1_bench(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_1x1_bench, self).__init__()
        self.main = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.main1 = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=1, stride=1, relu=False)
        )
        self.dim = out_channel
        self.norm = norm

    def forward(self, x):
        return self.main(x) + x + self.main1(x)

class ResBlock_do_fft_bench_cat_windows(nn.Module):
    def __init__(self, out_channel, win=64, norm='backward'):
        super(ResBlock_do_fft_bench_cat_windows, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft = nn.Sequential(
            BasicConv(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=False)
        )
        self.conv_cat = BasicConv(out_channel*2, out_channel, kernel_size=1, stride=1, relu=False)
        self.dim = out_channel
        self.norm = norm
        self.win=win
    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        x_windows, batch_list = window_partitionx(x, self.win)
        y = torch.fft.rfft2(x_windows, norm=self.norm)
        # y_imag = y.imag
        # y_real = y.real
        y_f = torch.cat([y.real, y.imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(self.win, self.win), norm=self.norm)
        y = window_reversex(y, self.win, H, W, batch_list)
        return self.conv_cat(torch.cat([self.main(x), y], dim=1)) + x
class ResBlock_do_fft_bench_cat_windows_shift(nn.Module):
    def __init__(self, out_channel, win=32, norm='backward'):
        super(ResBlock_do_fft_bench_cat_windows_shift, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft1 = fft_bench_x(out_channel)
        self.main_fft2 = fft_bench_x(out_channel)
        self.conv_cat = BasicConv(out_channel*2, out_channel, kernel_size=1, stride=1, relu=False)
        self.dim = out_channel
        self.norm = norm
        self.win=win
        self.shift_size = win // 2
    def forward(self, x):
        _, C, H, W = x.shape
        x1, x2 = torch.chunk(x, 2, dim=1)
        if self.shift_size != 0:
            x2_shift = F.pad(x2, (self.shift_size, 0, self.shift_size, 0), mode='constant')
            # torch.roll(x2, shifts=(-self.shift_size, -self.shift_size), dims=(-1, -2))
        else:
            x2_shift = x2
        x1_windows, batch_list1 = window_partitionx(x1, self.win)
        x2_windows, batch_list2 = window_partitionx(x2_shift, self.win)
        x1_windows = self.main_fft1(x1_windows)
        x2_windows = self.main_fft2(x2_windows)
        x1 = window_reversex(x1_windows, self.win, H, W, batch_list1)
        if self.shift_size > 0:
            x2_shift = window_reversex(x2_windows, self.win, H+self.shift_size, W+self.shift_size, batch_list2)
            x2 = x2_shift[:, :, self.shift_size:, self.shift_size:]
        else:
            x2 = window_reversex(x2_windows, self.win, H, W, batch_list2)
        return self.conv_cat(torch.cat([self.main(x), x1, x2], dim=1)) + x
class ResBlock_do_fft_bench_cat_multi_windows(nn.Module):
    def __init__(self, out_channel, win=[8, 16], norm='backward'):
        super(ResBlock_do_fft_bench_cat_multi_windows, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, groups=out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, groups=out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft1 = fft_bench(out_channel)
        self.main_fft2 = fft_bench(out_channel)
        # self.main_fft3 = fft_bench(out_channel)
        self.conv_cat = BasicConv(out_channel*3, out_channel, kernel_size=1, stride=1, relu=False)
        self.dim = out_channel
        self.norm = norm
        self.win = win
    def forward(self, x):
        _, _, H, W = x.shape
        # dim = 1
        x_windows1, batch_list1 = window_partitionx(x, self.win[0])
        x_windows2, batch_list2 = window_partitionx(x, self.win[1])
        # x_windows3, batch_list3 = window_partitionx(x, self.win[2])
        y1 = self.main_fft1(x_windows1)
        y2 = self.main_fft2(x_windows2)
        # y3 = self.main_fft3(x_windows3)

        y1 = window_reversex(y1, self.win[0], H, W, batch_list1)
        y2 = window_reversex(y2, self.win[1], H, W, batch_list2)
        # y3 = window_reversex(y3, self.win[2], H, W, batch_list3)
        return self.conv_cat(torch.cat([self.main(x), y1, y2], dim=1)) + x
class ResBlock_do_dct_bench_cat_multi_windows(nn.Module):
    def __init__(self, out_channel, win=[8, 16], norm='backward'):
        super(ResBlock_do_dct_bench_cat_multi_windows, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, groups=1, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, groups=1, kernel_size=3, stride=1, relu=False)
        )
        self.main_dct1 = dct_window_bench(out_channel, win[0])
        self.main_dct2 = dct_window_bench(out_channel, win[1])
        # self.main_fft3 = fft_bench(out_channel)
        self.conv_cat = BasicConv(out_channel*3, out_channel, kernel_size=1, stride=1, relu=False)
        self.dim = out_channel
        self.norm = norm
        self.win = win

    def forward(self, x):
        y1 = self.main_dct1(x)
        y2 = self.main_dct2(x)
        return self.conv_cat(torch.cat([self.main(x), y1, y2], dim=1)) + x
class ResBlock_do_fft_bench_cat(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_do_fft_bench_cat, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft = nn.Sequential(
            BasicConv(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=False)
        )
        self.conv_cat = BasicConv(out_channel*2, out_channel, kernel_size=1, stride=1, relu=False)
        self.dim = out_channel
        self.norm = norm
    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        # y_imag = y.imag
        # y_real = y.real
        y_f = torch.cat([y.real, y.imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.conv_cat(torch.cat([self.main(x), y], dim=1)) + x
class ResBlock_do_fft_bench_fine(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_do_fft_bench_fine, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft = nn.Sequential(
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=False)
        )
        self.dim = out_channel
        self.norm = norm
    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y



class ResBlock_do_fft_bench_complex_mlp(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_do_fft_bench_complex_mlp, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.act_fft = nn.ReLU()
        self.complex_weight1_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        self.dim = out_channel
        self.norm = norm
    def forward(self, x):
        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
        y = rearrange(y, 'b c h w -> b h w c')
        y = y @ weight1
        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.act_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        # y.real, y.imag = self.act_fft(y.real), self.act_fft(y.imag)
        # y = torch.cat([y.real, y.imag], dim=dim)
        # y = self.act_fft(y)
        # y_real, y_imag = torch.chunk(y, 2, dim=dim)
        # y = torch.complex(y_real, y_imag)
        y = y @ weight2
        y = rearrange(y, 'b h w c -> b c h w')
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y

class ResBlock_fft_bench_complex_mlp(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_fft_bench_complex_mlp, self).__init__()
        self.main = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.act_fft = nn.ReLU()
        self.complex_weight1_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        self.dim = out_channel
        self.norm = norm
    def forward(self, x):
        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
        y = rearrange(y, 'b c h w -> b h w c')
        y = y @ weight1
        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.act_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        # y.real, y.imag = self.act_fft(y.real), self.act_fft(y.imag)
        # y = torch.cat([y.real, y.imag], dim=dim)
        # y = self.act_fft(y)
        # y_real, y_imag = torch.chunk(y, 2, dim=dim)
        # y = torch.complex(y_real, y_imag)
        y = y @ weight2
        y = rearrange(y, 'b h w c -> b c h w')
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y
class ResBlock_window_mlp(nn.Module):
    def __init__(self, out_channel, win=32):
        super(ResBlock_window_mlp, self).__init__()
        self.win = win
        self.window_mlp = window_mlp(win*win)
        self.fft_mlp = fft_bench_complex_mlp(out_channel)

    def forward(self, x):
        _, _, H, W = x.shape
        x_windows, batch_list = window_partitionx(x, self.win)
        x_windows = self.window_mlp(x_windows)
        x_r = window_reversex(x_windows, self.win, H, W, batch_list)
        y = self.fft_mlp(x)
        return x + x_r + y
class ResBlock_window_fft_mlp(nn.Module):
    def __init__(self, out_channel, win_fft=32):
        super(ResBlock_window_fft_mlp, self).__init__()
        self.win_fft = win_fft
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.fft_mlp = fft_bench_complex_mlp(out_channel)

    def forward(self, x):
        _, _, H, W = x.shape
        y, batch_list_y = window_partitionx(x, self.win_fft)
        y = self.fft_mlp(y)
        y = window_reversex(y, self.win_fft, H, W, batch_list_y)
        return x + self.main(x) + y
class window_mlp(nn.Module):
    def __init__(self, out_channel):
        super(window_mlp, self).__init__()
        self.act = nn.ReLU()
        self.weight1 = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.weight2 = nn.Parameter(torch.Tensor(out_channel, out_channel))
        init.kaiming_uniform_(self.weight1, a=math.sqrt(16))
        init.kaiming_uniform_(self.weight2, a=math.sqrt(16))

    def forward(self, x):
        _, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = x @ self.weight1
        x = self.act(x)
        x = x @ self.weight2
        x = rearrange(x, 'b (c h w) -> b c h w', c=C, h=H, w=W)
        return x
class window_dcmlp(nn.Module):
    def __init__(self, out_channel, win):
        super(window_dcmlp, self).__init__()
        self.act = nn.ReLU()
        self.weight1 = nn.Parameter(torch.Tensor(win, win))
        self.conv = BasicConv_do(out_channel, out_channel, kernel_size=1, stride=1, relu=False)
        init.kaiming_uniform_(self.weight1, a=math.sqrt(16))

    def forward(self, x):
        _, _, H, W = x.shape
        x = rearrange(x, 'b c h w -> b c (h w)')
        x = x @ self.weight1
        x = self.act(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=H, w=W)
        x = self.conv(x)
        return x
class window_dmlp(nn.Module):
    def __init__(self, out_channel):
        super(window_dmlp, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.weight1 = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.weight2 = nn.Parameter(torch.Tensor(out_channel, out_channel))
        init.kaiming_uniform_(self.weight1, a=math.sqrt(16))
        init.kaiming_uniform_(self.weight2, a=math.sqrt(16))

    def forward(self, x):
        _, _, H, W = x.shape
        x = rearrange(x, 'b c h w -> b c (h w)')
        x = x @ self.weight1
        x = self.act(x)
        x = x @ self.weight2
        x = rearrange(x, 'b c (h w) -> b c h w', h=H, w=W)
        return x

class fft_bench_complex_mlp(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(fft_bench_complex_mlp, self).__init__()
        self.act_fft = nn.ReLU(inplace=True)
        self.complex_weight1_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        self.dim = out_channel
        self.norm = norm
        # self.min = inf
        # self.max = -inf
    def forward(self, x, save_dir=None):
        if save_dir is not None:
            _, _, H, W = x.shape
            y = torch.fft.rfft2(x, norm=self.norm)
            dim = 1
            weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
            weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
            y = rearrange(y, 'b c h w -> b h w c')
            y1 = y @ weight1
            y = torch.cat([y1.real, y1.imag], dim=dim)
            y2 = self.act_fft(y)
            y_real, y_imag = torch.chunk(y2, 2, dim=dim)
            y2 = torch.complex(y_real, y_imag)
            y = y2 @ weight2
            y = rearrange(y, 'b h w c -> b c h w')
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
            y1 = rearrange(y1, 'b h w c -> b c h w')
            y1 = torch.fft.irfft2(y1, s=(H, W), norm=self.norm)
            y2 = rearrange(y2, 'b h w c -> b c h w')
            y2 = torch.fft.irfft2(y2, s=(H, W), norm=self.norm)
            # in_dir = os.path.join(save_dir, 'in.npy')
            y1_dir = os.path.join(save_dir, 'fft1.npy')
            y2_dir = os.path.join(save_dir, 'fft_relu.npy')
            out_dir = os.path.join(save_dir, 'fft2.npy')
            # np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(y1_dir, y1.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(y2_dir, y2.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(out_dir, y.permute(0, 2, 3, 1).cpu().detach().numpy())
            return y
        else:
            _, _, H, W = x.shape
            y = torch.fft.rfft2(x, norm=self.norm)
            dim = 1
            weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
            weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
            y = rearrange(y, 'b c h w -> b h w c')
            y = y @ weight1
            y = torch.cat([y.real, y.imag], dim=dim)


            # self.min = min(self.min, np.percentile(y.cpu().numpy(), 95))
            # self.max = max(self.max, np.percentile(y.cpu().numpy(), 5))
            # print(self.max, self.min)
            y = self.act_fft(y)
            y_real, y_imag = torch.chunk(y, 2, dim=dim)
            y = torch.complex(y_real, y_imag)
            y = y @ weight2
            y = rearrange(y, 'b h w c -> b c h w')
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
            return y


class fft_bench_complex_mlp_onlyrelu(nn.Module):
    def __init__(self, out_channel=None, norm='backward', window_size=None):
        super(fft_bench_complex_mlp_onlyrelu, self).__init__()
        self.act_fft = nn.ReLU(inplace=True)
        # self.dim = out_channel
        self.norm = norm
        self.window_size = window_size

    def forward(self, x, save_dir=None):

        _, _, H, W = x.shape

        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.act_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)

        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class Attention_frelu_winx(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=256):
        super(Attention_frelu_winx, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.window_size = window_size
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.frelu = fft_bench_complex_mlp_onlyrelu()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, H, W = x.shape
        # _, _, H, W = x.shape

        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        qkv = self.qkv_dwconv(self.relu(self.qkv(x)))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.frelu(attn)
        # attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=self.window_size,
                        w=self.window_size)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            out = window_reversex(out, self.window_size, H, W, batch_list)

        out = self.project_out(out)
        return out





##########################################################################
class TransformerBlockx(nn.Module):
    def __init__(self, dim=32, num_heads=1, ffn_expansion_factor=1, bias=False, LayerNorm_type='WithBias)'):
        # def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlockx, self).__init__()

        # self.norm1 = LayerNorm(dim, LayerNorm_type)
        # self.attn = Attention(dim, num_heads, bias)

        self.attn = Attention_frelu_winx(dim, num_heads, bias, window_size=64)
        # self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, save_dir=None):
        x = x + self.attn(x)
        # x = x + self.ffn(self.norm2(x))
        return x
class fft_bench_complex_mlp_onlyrelu_silu(nn.Module):
    def __init__(self, out_channel, norm='backward', window_size=None):
        super(fft_bench_complex_mlp_onlyrelu_silu, self).__init__()
        self.act_fft = SiLU()
        self.dim = out_channel
        self.norm = norm
        self.window_size = window_size

    def forward(self, x, save_dir=None):

        _, _, H, W = x.shape

        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.act_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)

        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class fft_bench_complex_mlp_onlyrelu_clamp(nn.Module):
    def __init__(self, out_channel, norm='backward', window_size=None, act_min=-inf, act_max=inf):
        super(fft_bench_complex_mlp_onlyrelu_clamp, self).__init__()
        # self.act_fft = nn.ReLU(inplace=True)
        self.dim = out_channel
        self.norm = norm
        self.window_size = window_size
        self.act_clamp_min = act_min
        self.act_clamp_max = act_max

    def forward(self, x, save_dir=None):

        _, _, H, W = x.shape

        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        y = torch.cat([y.real, y.imag], dim=dim)

        y = torch.clamp(y, min=self.act_clamp_min, max=self.act_clamp_max)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)

        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            y = torch.fft.irfft2(y, s=(self.window_size, self.window_size), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class fft_bench_complex_mlp_justrelu(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(fft_bench_complex_mlp_justrelu, self).__init__()
        self.act_fft = nn.ReLU(inplace=True)
        self.complex_weight1_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))

        # self.dim = out_channel
        self.norm = norm
    def forward(self, x, save_dir=None):
        if save_dir is not None:
            _, _, H, W = x.shape
            y = torch.fft.rfft2(x, norm=self.norm)
            dim = 1
            weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)

            y = rearrange(y, 'b c h w -> b h w c')
            y1 = y @ weight1
            y = torch.cat([y1.real, y1.imag], dim=dim)
            y2 = self.act_fft(y)
            y_real, y_imag = torch.chunk(y2, 2, dim=dim)
            y2 = torch.complex(y_real, y_imag)
            y1 = rearrange(y1, 'b h w c -> b c h w')
            y1 = torch.fft.irfft2(y1, s=(H, W), norm=self.norm)
            y2 = rearrange(y2, 'b h w c -> b c h w')
            y2 = torch.fft.irfft2(y2, s=(H, W), norm=self.norm)
            # in_dir = os.path.join(save_dir, 'in.npy')
            y1_dir = os.path.join(save_dir, 'fft1.npy')
            y2_dir = os.path.join(save_dir, 'fft_relu.npy')

            # np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(y1_dir, y1.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(y2_dir, y2.permute(0, 2, 3, 1).cpu().detach().numpy())

            return y2
        else:
            _, _, H, W = x.shape
            y = torch.fft.rfft2(x, norm=self.norm)
            dim = 1
            weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
            y = rearrange(y, 'b c h w -> b h w c')
            y = y @ weight1
            y = torch.cat([y.real, y.imag], dim=dim)
            y = self.act_fft(y)
            y_real, y_imag = torch.chunk(y, 2, dim=dim)
            y = torch.complex(y_real, y_imag)
            y = rearrange(y, 'b h w c -> b c h w')
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
            return y
class fft_bench_complex_mlp_norelu(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(fft_bench_complex_mlp_norelu, self).__init__()
        self.act_fft = nn.ReLU(inplace=True)
        self.complex_weight1_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))

        self.dim = out_channel
        self.norm = norm
    def forward(self, x, save_dir=None):
        if save_dir is not None:
            _, _, H, W = x.shape
            y = torch.fft.rfft2(x, norm=self.norm)
            dim = 1
            weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
            y = rearrange(y, 'b c h w -> b h w c')
            y1 = y @ weight1

            y1 = rearrange(y1, 'b h w c -> b c h w')
            y1 = torch.fft.irfft2(y1, s=(H, W), norm=self.norm)

            # in_dir = os.path.join(save_dir, 'in.npy')
            y1_dir = os.path.join(save_dir, 'fft1.npy')
            # np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(y1_dir, y1.permute(0, 2, 3, 1).cpu().detach().numpy())

            return y1
        else:
            _, _, H, W = x.shape
            y = torch.fft.rfft2(x, norm=self.norm)
            dim = 1
            weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
            y = rearrange(y, 'b c h w -> b h w c')
            y = y @ weight1
            y = rearrange(y, 'b h w c -> b c h w')
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
            return y
class fft_bench_complex_mlp_norelu2(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(fft_bench_complex_mlp_norelu2, self).__init__()
        self.complex_weight1_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))

        self.dim = out_channel
        self.norm = norm
    def forward(self, x, save_dir=None):

        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
        y = rearrange(y, 'b c h w -> b h w c')
        y = y @ weight1
        y = y @ weight2
        y = rearrange(y, 'b h w c -> b c h w')
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class res_fft_bench_complex_mlp(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(res_fft_bench_complex_mlp, self).__init__()
        self.act_fft = nn.ReLU(inplace=True)
        self.complex_weight1_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        self.dim = out_channel
        self.norm = norm
    def forward(self, x, save_dir=None):
        if save_dir is not None:
            _, _, H, W = x.shape
            y = torch.fft.rfft2(x, norm=self.norm)
            dim = 1
            weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
            weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
            y = rearrange(y, 'b c h w -> b h w c')
            y1 = y @ weight1
            y = torch.cat([y1.real, y1.imag], dim=dim)
            y2 = self.act_fft(y)
            y_real, y_imag = torch.chunk(y2, 2, dim=dim)
            y2 = torch.complex(y_real, y_imag)
            y = y2 @ weight2
            y = rearrange(y, 'b h w c -> b c h w')
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
            y1 = rearrange(y1, 'b h w c -> b c h w')
            y1 = torch.fft.irfft2(y1, s=(H, W), norm=self.norm)
            y2 = rearrange(y2, 'b h w c -> b c h w')
            y2 = torch.fft.irfft2(y2, s=(H, W), norm=self.norm)
            out = y + x
            in_dir = os.path.join(save_dir, 'in.npy')
            y1_dir = os.path.join(save_dir, 'fft1.npy')
            y2_dir = os.path.join(save_dir, 'fft_relu.npy')
            y3_dir = os.path.join(save_dir, 'fft2.npy')
            out_dir = os.path.join(save_dir, 'out.npy')
            np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(y1_dir, y1.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(y2_dir, y2.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(y3_dir, y.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(out_dir, out.permute(0, 2, 3, 1).cpu().detach().numpy())
            return out
        else:
            _, _, H, W = x.shape
            y = torch.fft.rfft2(x, norm=self.norm)
            dim = 1
            weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
            weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
            y = rearrange(y, 'b c h w -> b h w c')
            y = y @ weight1
            y = torch.cat([y.real, y.imag], dim=dim)
            y = self.act_fft(y)
            y_real, y_imag = torch.chunk(y, 2, dim=dim)
            y = torch.complex(y_real, y_imag)
            y = y @ weight2
            y = rearrange(y, 'b h w c -> b c h w')
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
            return y + x


def generate_center_mask(H, W, r, ir=False):

    mask = torch.zeros([H, W])

    for x in range(H):
        for y in range(W):
            tmp = (x - H//2) ** 2 + (y - W//2) ** 2
            # print(tmp, x, y)
            if tmp <= r ** 2:
                mask[x, y] = 1.
            else:
                continue
    if ir:
        mask = torch.ones_like(mask) - mask
    return mask

def generate_center_mask_rfft(H, W, r, ir=False):
    print(r, ir)
    mask_ = generate_center_mask(H, W, r, ir=ir)
    w = W//2+1
    mask = torch.zeros([H, w])
    mask[:H // 2, :] = mask_[-H // 2:, -w:]
    mask[-H // 2:, :] = mask_[:H // 2, -w:]
    return mask
class get_center_mask_rfft(nn.Module):
    def __init__(self, H=256, W=256, ir=False, inference=False):
        super(get_center_mask_rfft, self).__init__()
        # self.mask_ = generate_center_mask(H, W, r, ir=ir)
        self.H = H
        self.W = W
        self.ir = ir
        self.HW = np.int(np.sqrt(H*H + W*W)+1)
        self.x = nn.Parameter(torch.Tensor(0))
        # self.inference = inference

    def forward(self, ):
        x = torch.sigmoid(self.x) * torch.tensor(self.HW, device=self.x.device)
        # x = torch.clamp(x, 0, self.HW+1)
        mask_ = generate_center_mask(self.H, self.W, x, ir=self.ir)
        w = self.W // 2 + 1
        mask = torch.zeros([self.H, w])
        mask[:self.H // 2, :] = mask_[-self.H // 2:, -w:]
        mask[-self.H // 2:, :] = mask_[self.H // 2:, -w:]
        return mask
class fft_bench_complex_mlp_low_freq(nn.Module):
    def __init__(self, out_channel, norm='backward', win=256):
        super(fft_bench_complex_mlp_low_freq, self).__init__()
        self.act_fft = nn.ReLU(inplace=True)
        self.complex_weight1_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        self.dim = out_channel
        self.norm = norm
        self.mask = generate_center_mask(win, win, win//16, ir=False)
    def forward(self, x, save_dir=None):
        if save_dir is not None:
            _, _, H, W = x.shape
            y = torch.fft.fft2(x, norm=self.norm)
            y = torch.fft.fftshift(y)
            mask = torch.complex(self.mask.to(y.device), torch.zeros_like(self.mask.to(y.device)))
            y = y * mask
            dim = 1
            weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
            weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
            y = rearrange(y, 'b c h w -> b h w c')
            y1 = y @ weight1
            y = torch.cat([y1.real, y1.imag], dim=dim)
            y2 = self.act_fft(y)
            y_real, y_imag = torch.chunk(y2, 2, dim=dim)
            y2 = torch.complex(y_real, y_imag)
            y = y2 @ weight2
            y = rearrange(y, 'b h w c -> b c h w')
            y = torch.fft.ifftshift(y)
            y = torch.fft.ifft2(y, s=(H, W), norm=self.norm)
            # print(y.shape, y.dtype)
            y1 = rearrange(y1, 'b h w c -> b c h w')
            y1 = torch.fft.ifftshift(y1)
            y1 = torch.fft.ifft2(y1, s=(H, W), norm=self.norm)
            y2 = rearrange(y2, 'b h w c -> b c h w')
            y2 = torch.fft.ifftshift(y2)
            y2 = torch.fft.ifft2(y2, s=(H, W), norm=self.norm)
            y1_dir = os.path.join(save_dir, 'fft1.npy')
            y2_dir = os.path.join(save_dir, 'fft_relu.npy')
            out_dir = os.path.join(save_dir, 'fft2.npy')
            np.save(out_dir, y.real.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(y1_dir, y1.real.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(y2_dir, y2.real.permute(0, 2, 3, 1).cpu().detach().numpy())
            return y.real
        else:
            _, _, H, W = x.shape
            y = torch.fft.fft2(x, norm=self.norm)
            y = torch.fft.fftshift(y)
            mask = torch.complex(self.mask.to(y.device), torch.zeros_like(self.mask.to(y.device)))
            y = y * mask
            dim = 1
            weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
            weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
            y = rearrange(y, 'b c h w -> b h w c')
            y = y @ weight1
            y = torch.cat([y.real, y.imag], dim=dim)
            y = self.act_fft(y)
            y_real, y_imag = torch.chunk(y, 2, dim=dim)
            y = torch.complex(y_real, y_imag)
            y = y @ weight2
            y = rearrange(y, 'b h w c -> b c h w')
            y = torch.fft.ifftshift(y)
            y = torch.fft.ifft2(y, s=(H, W), norm=self.norm)
            # print(y.shape, y.dtype)
            return y.real
class fft_bench_complex_mlp_high_freq_r(nn.Module):
    def __init__(self, out_channel, norm='backward', win=256, r=256//8):
        super(fft_bench_complex_mlp_high_freq_r, self).__init__()
        self.act_fft = nn.ReLU(inplace=True)
        self.complex_weight1_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        self.dim = out_channel
        self.norm = norm
        self.mask = generate_center_mask_rfft(win, win, r, ir=True)
    def forward(self, x, save_dir=None):
        if save_dir is not None:
            _, _, H, W = x.shape
            y = torch.fft.rfft2(x, norm=self.norm)

            mask = torch.complex(self.mask.to(y.device), torch.zeros_like(self.mask.to(y.device)))
            y = y * mask
            dim = 1
            weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
            weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
            y = rearrange(y, 'b c h w -> b h w c')
            y1 = y @ weight1
            y = torch.cat([y1.real, y1.imag], dim=dim)
            y2 = self.act_fft(y)
            y_real, y_imag = torch.chunk(y2, 2, dim=dim)
            y2 = torch.complex(y_real, y_imag)
            y = y2 @ weight2
            y = rearrange(y, 'b h w c -> b c h w')
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
            # print(y.shape, y.dtype)
            y1 = rearrange(y1, 'b h w c -> b c h w')
            y1 = torch.fft.irfft2(y1, s=(H, W), norm=self.norm)
            y2 = rearrange(y2, 'b h w c -> b c h w')

            y2 = torch.fft.irfft2(y2, s=(H, W), norm=self.norm)
            y1_dir = os.path.join(save_dir, 'fft1.npy')
            y2_dir = os.path.join(save_dir, 'fft_relu.npy')
            out_dir = os.path.join(save_dir, 'fft2.npy')
            np.save(out_dir, y.real.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(y1_dir, y1.real.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(y2_dir, y2.real.permute(0, 2, 3, 1).cpu().detach().numpy())
            return y
        else:
            _, _, H, W = x.shape
            y = torch.fft.rfft2(x, norm=self.norm)
            mask = torch.complex(self.mask.to(y.device), torch.zeros_like(self.mask.to(y.device)))
            y = y * mask
            dim = 1
            weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
            weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
            y = rearrange(y, 'b c h w -> b h w c')
            y = y @ weight1
            y = torch.cat([y.real, y.imag], dim=dim)
            y = self.act_fft(y)
            y_real, y_imag = torch.chunk(y, 2, dim=dim)
            y = torch.complex(y_real, y_imag)
            y = y @ weight2
            y = rearrange(y, 'b h w c -> b c h w')
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
            # print(y.shape, y.dtype)
            return y
class act_high_freq_r(nn.Module):
    def __init__(self, out_channel, norm='backward', win=256, r=256//8):
        super(act_high_freq_r, self).__init__()
        self.norm = norm
        self.mask = generate_center_mask_rfft(win, win, r, ir=True)

    def forward(self, x, save_dir=None):

        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm=self.norm)
        mask = torch.complex(self.mask.to(y.device), torch.zeros_like(self.mask.to(y.device)))
        y = y * mask
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        # print(y.shape, y.dtype)
        return y


class act_low_freq_r(nn.Module):
    def __init__(self, out_channel, norm='backward', win=256, r=256 // 8):
        super(act_low_freq_r, self).__init__()
        self.norm = norm
        self.mask = generate_center_mask_rfft(win, win, r, ir=False)

    def forward(self, x, save_dir=None):
        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm=self.norm)
        mask = torch.complex(self.mask.to(y.device), torch.zeros_like(self.mask.to(y.device)))
        y = y * mask
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        # print(y.shape, y.dtype)
        return y
class fft_bench_complex_mlp_low_freq_r(nn.Module):
    def __init__(self, out_channel, norm='backward', win=256, r=256 // 8):
        super(fft_bench_complex_mlp_low_freq_r, self).__init__()
        self.act_fft = nn.ReLU(inplace=True)
        self.complex_weight1_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        self.dim = out_channel
        self.norm = norm
        self.mask = generate_center_mask_rfft(win, win, r, ir=False)

    def forward(self, x, save_dir=None):
        if save_dir is not None:
            _, _, H, W = x.shape
            y = torch.fft.rfft2(x, norm=self.norm)

            mask = torch.complex(self.mask.to(y.device), torch.zeros_like(self.mask.to(y.device)))
            y = y * mask
            dim = 1
            weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
            weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
            y = rearrange(y, 'b c h w -> b h w c')
            y1 = y @ weight1
            y = torch.cat([y1.real, y1.imag], dim=dim)
            y2 = self.act_fft(y)
            y_real, y_imag = torch.chunk(y2, 2, dim=dim)
            y2 = torch.complex(y_real, y_imag)
            y = y2 @ weight2
            y = rearrange(y, 'b h w c -> b c h w')
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
            # print(y.shape, y.dtype)
            y1 = rearrange(y1, 'b h w c -> b c h w')
            y1 = torch.fft.irfft2(y1, s=(H, W), norm=self.norm)
            y2 = rearrange(y2, 'b h w c -> b c h w')

            y2 = torch.fft.irfft2(y2, s=(H, W), norm=self.norm)
            y1_dir = os.path.join(save_dir, 'fft1.npy')
            y2_dir = os.path.join(save_dir, 'fft_relu.npy')
            out_dir = os.path.join(save_dir, 'fft2.npy')
            np.save(out_dir, y.real.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(y1_dir, y1.real.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(y2_dir, y2.real.permute(0, 2, 3, 1).cpu().detach().numpy())
            return y
        else:
            _, _, H, W = x.shape
            y = torch.fft.rfft2(x, norm=self.norm)
            mask = torch.complex(self.mask.to(y.device), torch.zeros_like(self.mask.to(y.device)))
            y = y * mask
            dim = 1
            weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
            weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
            y = rearrange(y, 'b c h w -> b h w c')
            y = y @ weight1
            y = torch.cat([y.real, y.imag], dim=dim)
            y = self.act_fft(y)
            y_real, y_imag = torch.chunk(y, 2, dim=dim)
            y = torch.complex(y_real, y_imag)
            y = y @ weight2
            y = rearrange(y, 'b h w c -> b c h w')
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
            # print(y.shape, y.dtype)
            return y
class fft_bench_complex_mlp_high_freq(nn.Module):
    def __init__(self, out_channel, norm='backward', win=256):
        super(fft_bench_complex_mlp_high_freq, self).__init__()
        self.act_fft = nn.ReLU(inplace=True)
        self.complex_weight1_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        self.dim = out_channel
        self.norm = norm
        self.mask = generate_center_mask(win, win, win//16, ir=True)
    def forward(self, x, save_dir=None):
        if save_dir is not None:
            _, _, H, W = x.shape
            y = torch.fft.fft2(x, norm=self.norm)
            y = torch.fft.fftshift(y)
            mask = torch.complex(self.mask.to(y.device), torch.zeros_like(self.mask.to(y.device)))
            y = y * mask
            dim = 1
            weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
            weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
            y = rearrange(y, 'b c h w -> b h w c')
            y1 = y @ weight1
            y = torch.cat([y1.real, y1.imag], dim=dim)
            y2 = self.act_fft(y)
            y_real, y_imag = torch.chunk(y2, 2, dim=dim)
            y2 = torch.complex(y_real, y_imag)
            y = y2 @ weight2
            y = rearrange(y, 'b h w c -> b c h w')
            y = torch.fft.ifftshift(y)
            y = torch.fft.ifft2(y, s=(H, W), norm=self.norm)
            # print(y.shape, y.dtype)
            y1 = rearrange(y1, 'b h w c -> b c h w')
            y1 = torch.fft.ifftshift(y1)
            y1 = torch.fft.ifft2(y1, s=(H, W), norm=self.norm)
            y2 = rearrange(y2, 'b h w c -> b c h w')
            y2 = torch.fft.ifftshift(y2)
            y2 = torch.fft.ifft2(y2, s=(H, W), norm=self.norm)
            y1_dir = os.path.join(save_dir, 'fft1.npy')
            y2_dir = os.path.join(save_dir, 'fft_relu.npy')
            out_dir = os.path.join(save_dir, 'fft2.npy')
            np.save(out_dir, y.real.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(y1_dir, y1.real.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(y2_dir, y2.real.permute(0, 2, 3, 1).cpu().detach().numpy())
            return y.real
        else:
            _, _, H, W = x.shape
            y = torch.fft.fft2(x, norm=self.norm)
            y = torch.fft.fftshift(y)
            mask = torch.complex(self.mask.to(y.device), torch.zeros_like(self.mask.to(y.device)))
            y = y * mask
            dim = 1
            weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
            weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
            y = rearrange(y, 'b c h w -> b h w c')
            y = y @ weight1
            y = torch.cat([y.real, y.imag], dim=dim)
            y = self.act_fft(y)
            y_real, y_imag = torch.chunk(y, 2, dim=dim)
            y = torch.complex(y_real, y_imag)
            y = y @ weight2
            y = rearrange(y, 'b h w c -> b c h w')
            y = torch.fft.ifftshift(y)
            y = torch.fft.ifft2(y, s=(H, W), norm=self.norm)
            # print(y.shape, y.dtype)
            return y.real
class fft_bench_complex_mlp_high_low_freq(nn.Module):
    def __init__(self, out_channel, norm='backward', win=256):
        super(fft_bench_complex_mlp_high_low_freq, self).__init__()
        self.act_fft = nn.ReLU(inplace=True)
        self.complex_weight1_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight1_real2 = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight1_imag2 = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_real2 = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_imag2 = nn.Parameter(torch.Tensor(out_channel, out_channel))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_real2, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag2, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real2, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag2, a=math.sqrt(16))
        self.dim = out_channel
        self.norm = norm
        self.mask_low = generate_center_mask_rfft(win, win, win // 4, ir=False)
        self.mask_high = generate_center_mask_rfft(win, win, win//4, ir=True)
    def forward(self, x, save_dir=None):

        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm=self.norm)
        mask_high = torch.complex(self.mask_high.to(y.device), torch.zeros_like(self.mask_high.to(y.device)))
        mask_low = torch.complex(self.mask_low.to(y.device), torch.zeros_like(self.mask_low.to(y.device)))
        y_high = y * mask_high
        y_low = y * mask_low
        dim = 1
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
        weight12 = torch.complex(self.complex_weight1_real2, self.complex_weight1_imag2)
        weight22 = torch.complex(self.complex_weight2_real2, self.complex_weight2_imag2)
        y_high = rearrange(y_high, 'b c h w -> b h w c')
        y_high = y_high @ weight1
        y_high = torch.cat([y_high.real, y_high.imag], dim=dim)
        y_high = self.act_fft(y_high)
        y_high_real, y_high_imag = torch.chunk(y_high, 2, dim=dim)
        y_high = torch.complex(y_high_real, y_high_imag)
        y_high = y_high @ weight2
        y_low = rearrange(y_low, 'b c h w -> b h w c')
        y_low = y_low @ weight12
        y_low = torch.cat([y_low.real, y_low.imag], dim=dim)
        y_low = self.act_fft(y_low)
        y_low_real, y_low_imag = torch.chunk(y_low, 2, dim=dim)
        y_low = torch.complex(y_low_real, y_low_imag)
        y_low = y_low @ weight22
        y = y_high + y_low
        y = rearrange(y, 'b h w c -> b c h w')
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        # print(y.shape, y.dtype)
        return y
class fft_bench_complex_mlp_1x1(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(fft_bench_complex_mlp_1x1, self).__init__()
        self.act_fft = nn.ReLU(inplace=True)
        self.complex_weight1_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        self.conv1 = nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.dim = out_channel
        self.norm = norm
    def forward(self, x):
        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm=self.norm)
        dim = 1
        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        y = rearrange(y, 'b c h w -> b h w c')
        y = y @ weight1
        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.act_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = rearrange(y, 'b h w c -> b c h w')
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        y = self.conv1(y)
        return y

class fft_bench_real_mlp(nn.Module):
    def __init__(self, out_channel, norm='backward', return_2=False):
        super(fft_bench_real_mlp, self).__init__()
        self.act_fft = nn.ReLU(inplace=True)
        self.complex_weight1_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        self.dim = out_channel
        self.norm = norm
        self.return_2 = return_2

    def forward(self, x):
        if self.return_2:
            _, _, H, W = x.shape
            y = torch.fft.rfft2(x, norm=self.norm)
            dim = 1
            weight1 = torch.complex(self.complex_weight1_real, torch.zeros_like(self.complex_weight1_real))
            weight2 = torch.complex(self.complex_weight2_real, torch.zeros_like(self.complex_weight2_real))
            y = rearrange(y, 'b c h w -> b h w c')
            y1 = y @ weight1
            y = torch.cat([y1.real, y1.imag], dim=dim)
            y = self.act_fft(y)
            y_real, y_imag = torch.chunk(y, 2, dim=dim)
            y2 = torch.complex(y_real, y_imag)
            y = y2 @ weight2
            y = rearrange(y, 'b h w c -> b c h w')
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
            y1 = rearrange(y1, 'b h w c -> b c h w')
            y1 = torch.fft.irfft2(y1, s=(H, W), norm=self.norm)
            y2 = rearrange(y2, 'b h w c -> b c h w')
            y2 = torch.fft.irfft2(y2, s=(H, W), norm=self.norm)
            return y1, y2, y
        else:
            _, _, H, W = x.shape
            y = torch.fft.rfft2(x, norm=self.norm)
            dim = 1
            weight1 = torch.complex(self.complex_weight1_real, torch.zeros_like(self.complex_weight1_real))
            weight2 = torch.complex(self.complex_weight2_real, torch.zeros_like(self.complex_weight2_real))
            y = rearrange(y, 'b c h w -> b h w c')
            y = y @ weight1
            y = torch.cat([y.real, y.imag], dim=dim)
            y = self.act_fft(y)
            y_real, y_imag = torch.chunk(y, 2, dim=dim)
            y = torch.complex(y_real, y_imag)
            y = y @ weight2
            y = rearrange(y, 'b h w c -> b c h w')
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
            return y

# class ResBlock_fft_bench_complex_mlp(nn.Module):
#     def __init__(self, out_channel, norm='backward'):
#         super(ResBlock_fft_bench_complex_mlp, self).__init__()
#         self.main = nn.Sequential(
#             BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
#             BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
#         )
#         self.act_fft = nn.ReLU()
#         self.complex_weight1 = nn.Parameter(torch.Tensor(out_channel, out_channel, 2)) # (torch.rand(out_channel, out_channel, 2, dtype=torch.float32) * 0.01)
#
#         self.complex_weight2 = nn.Parameter(torch.Tensor(out_channel, out_channel, 2)) # (torch.rand(out_channel, out_channel, 2, dtype=torch.float32) * 0.01)
#         init.kaiming_uniform_(self.complex_weight1, a=math.sqrt(16))
#         init.kaiming_uniform_(self.complex_weight2, a=math.sqrt(16))
#         self.norm = norm
#     def forward(self, x):
#         _, _, H, W = x.shape
#         y = torch.fft.rfft2(x, norm=self.norm)
#         weight1 = torch.view_as_complex(self.complex_weight1)
#         weight2 = torch.view_as_complex(self.complex_weight2)
#         y = rearrange(y, 'b c h w -> b h w c')
#         y = y @ weight1
#         y.real, y.imag = self.act_fft(y.real), self.act_fft(y.imag)
#         y = y @ weight2
#         y = rearrange(y, 'b h w c -> b c h w')
#         y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
#         return self.main(x) + x + y


class ResBlock_do_fft_bench_mean(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_do_fft_bench_mean, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft = nn.Sequential(
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=True)
        )
        self.conv = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=1, stride=1, relu=False)
        )
        self.dim = out_channel
        self.norm = norm
    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        # y = torch.mean(y, dim=1, keepdim=True)
        return self.main(x) + x + self.conv(y)

class ResBlock_do_fft_bench_dc(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_do_fft_bench_dc, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True, groups=out_channel),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False, groups=out_channel)
        )
        self.main_fft = nn.Sequential(
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=False)
        )
        self.dim = out_channel
        self.norm = norm
    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y

class ResBlock_fft_bench(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResBlock_fft_bench, self).__init__()
        self.main = nn.Sequential(
            BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=True),
            BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft = nn.Sequential(
            BasicConv(n_feat*2, n_feat*2, kernel_size=1, stride=1, relu=True),
            BasicConv(n_feat*2, n_feat*2, kernel_size=1, stride=1, relu=False)
        )
        self.dim = n_feat
        self.norm = norm
    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        # y = torch.fft.rfft2(x, norm='ortho')
        y = torch.fft.rfft2(x, norm=self.norm)
        y_f = torch.cat([y.real, y.imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm='ortho')
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + y + x

class ResBlock_fft_bench_test(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResBlock_fft_bench_test, self).__init__()
        self.main = conv_bench(n_feat)#
        self.main_fft = fft_bench_coord(n_feat, norm)
        # self.dropout = nn.Dropout2d(p=0.5)
    def forward(self, x):
        y = self.main_fft(x)
        # y = self.dropout(y)
        return self.main(x) + y + x


class ResBlock_fft_bench_silu(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResBlock_fft_bench_silu, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = fft_bench(n_feat, norm, act_method=SiLU)

    def forward(self, x):
        return self.main(x) + self.main_fft(x) + x
class ResBlock_conv_fftconv_bench(nn.Module):
    def __init__(self, n_feat):
        super(ResBlock_conv_fftconv_bench, self).__init__()
        self.conv = conv_bench(n_feat)

    def forward(self, x):
        return x + self.conv(x)
class ResBlock_fft_bench_test_nox(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResBlock_fft_bench_test_nox, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = fft_bench_phase(n_feat, norm)

    def forward(self, x):
        return self.main(x) + self.main_fft(x)
class ResFourier_complex_calyer(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_calyer, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = fft_bench_complex_mlp(n_feat, norm)
        self.calyer = simam_module()

    def forward(self, x):
        res = self.main(x) + self.main_fft(x)
        res = self.calyer(res)
        # res += x
        return res + x
class ResBlock_save(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock_save, self).__init__()
        self.main = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True, norm=False),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False, norm=False)
        )
    def forward(self, x, save_dir=None):
        if save_dir is not None:
            res = self.main(x)
            in_dir = os.path.join(save_dir, 'in.npy')
            res_dir = os.path.join(save_dir, 'res.npy')
            out_dir = os.path.join(save_dir, 'out.npy')
            out = res + x
            np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(res_dir, res.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(out_dir, out.permute(0, 2, 3, 1).cpu().detach().numpy())
            return out
        else:
            res = self.main(x)
            # res += x
            return res + x
class ResBlock_1x1(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock_1x1, self).__init__()
        self.main = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True, norm=False),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False, norm=False)
        )
        self.main1 = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=1, stride=1, relu=True, norm=False),
            BasicConv(out_channel, out_channel, kernel_size=1, stride=1, relu=False, norm=False)
        )
    def forward(self, x, save_dir=None):
        if save_dir is not None:
            res = self.main(x) + self.main1(x)
            in_dir = os.path.join(save_dir, 'in.npy')
            res_dir = os.path.join(save_dir, 'res.npy')
            out_dir = os.path.join(save_dir, 'out.npy')
            out = res + x
            np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(res_dir, res.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(out_dir, out.permute(0, 2, 3, 1).cpu().detach().numpy())
            return out
        else:
            res = self.main(x) + self.main1(x)
            # res += x
            return res + x
class ResFourier_1x1_f_1x1(nn.Module):
    def __init__(self, out_channel):
        super(ResFourier_1x1_f_1x1, self).__init__()
        self.main = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True, norm=False),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False, norm=False)
        )
        self.main1 = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=1, stride=1, relu=False, norm=False),
            fft_bench_complex_mlp_onlyrelu(out_channel),
            BasicConv(out_channel, out_channel, kernel_size=1, stride=1, relu=False, norm=False)
        )
    def forward(self, x, save_dir=None):

        res = self.main(x) + self.main1(x)
        # res += x
        return res + x
class ResFourier_complex(nn.Module):
    def __init__(self, n_feat, kernel_size=3, norm='backward'): # 'backward'
        super(ResFourier_complex, self).__init__()
        self.main = conv_bench(n_feat, kernel_size=kernel_size)
        self.main_fft = fft_bench_complex_mlp(n_feat, norm)

    def forward(self, x, save_dir=None):
        if save_dir is not None:
            conv = self.main(x)
            fft = self.main_fft(x, save_dir)
            res = conv + fft
            in_dir = os.path.join(save_dir, 'in.npy')
            res_dir = os.path.join(save_dir, 'res.npy')
            conv_dir = os.path.join(save_dir, 'conv.npy')
            # fft_dir = os.path.join(save_dir, 'fft.npy')
            out_dir = os.path.join(save_dir, 'out.npy')
            out = res + x
            np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(res_dir, res.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(conv_dir, conv.permute(0, 2, 3, 1).cpu().detach().numpy())
            # np.save(fft_dir, fft.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(out_dir, out.permute(0, 2, 3, 1).cpu().detach().numpy())
            return out
        else:
            res = self.main(x) + self.main_fft(x)
            # res += x
            return res + x
class ResFourier_complex_forward(nn.Module):
    def __init__(self, n_feat, kernel_size=3, norm='backward'): # 'backward'
        super(ResFourier_complex_forward, self).__init__()
        self.main = conv_bench(n_feat, kernel_size=kernel_size)
        self.main_fft = fft_bench_complex_mlp(n_feat, norm)

    def forward(self, x, save_dir=None):

        res = self.main_fft(self.main(x))
        # res += x
        return res + x
class ResFourier_complex_forward2(nn.Module):
    def __init__(self, n_feat, kernel_size=3, norm='backward'): # 'backward'
        super(ResFourier_complex_forward2, self).__init__()
        self.main = conv_bench(n_feat, kernel_size=kernel_size)
        self.main_fft = fft_bench_complex_mlp(n_feat, norm)

    def forward(self, x, save_dir=None):

        res = self.main(x)+x
        res = self.main_fft(res) + res
        # res += x
        return res
class ResFourier_complex_nores(nn.Module):
    def __init__(self, n_feat, kernel_size=3, norm='backward'): # 'backward'
        super(ResFourier_complex_nores, self).__init__()
        self.main = conv_bench(n_feat, kernel_size=kernel_size)
        self.main_fft = fft_bench_complex_mlp(n_feat, norm)

    def forward(self, x, save_dir=None):

        res = self.main(x) + self.main_fft(x)
        # res += x
        return res
class ResFourier_complex_justrelu(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_justrelu, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = fft_bench_complex_mlp_justrelu(n_feat, norm)

    def forward(self, x, save_dir=None):
        if save_dir is not None:
            conv = self.main(x)
            fft = self.main_fft(x, save_dir)
            res = conv + fft
            in_dir = os.path.join(save_dir, 'in.npy')
            res_dir = os.path.join(save_dir, 'res.npy')
            conv_dir = os.path.join(save_dir, 'conv.npy')
            # fft_dir = os.path.join(save_dir, 'fft.npy')
            out_dir = os.path.join(save_dir, 'out.npy')
            out = res + x
            np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(res_dir, res.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(conv_dir, conv.permute(0, 2, 3, 1).cpu().detach().numpy())
            # np.save(fft_dir, fft.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(out_dir, out.permute(0, 2, 3, 1).cpu().detach().numpy())
            return out
        else:
            res = self.main(x) + self.main_fft(x)
            # res += x
            return res + x
class ResFourier_complex_onlyrelu(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_onlyrelu, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = fft_bench_complex_mlp_onlyrelu(n_feat, norm)

    def forward(self, x, save_dir=None):
        if save_dir is not None:
            conv = self.main(x)
            fft = self.main_fft(x, save_dir)
            res = conv + fft
            in_dir = os.path.join(save_dir, 'in.npy')
            res_dir = os.path.join(save_dir, 'res.npy')
            conv_dir = os.path.join(save_dir, 'conv.npy')
            # fft_dir = os.path.join(save_dir, 'fft.npy')
            out_dir = os.path.join(save_dir, 'out.npy')
            out = res + x
            np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(res_dir, res.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(conv_dir, conv.permute(0, 2, 3, 1).cpu().detach().numpy())
            # np.save(fft_dir, fft.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(out_dir, out.permute(0, 2, 3, 1).cpu().detach().numpy())
            return out
        else:
            res = self.main(x) + self.main_fft(x)
            # res += x
            return res + x
class ResFourier_complex_onlyrelu2(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_onlyrelu2, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = fft_bench_complex_mlp_onlyrelu(n_feat, norm, window_size=None)

    def forward(self, x, save_dir=None):
        if save_dir is not None:
            conv = self.main(x)
            fft = self.main_fft(x, save_dir)
            res = conv + fft
            in_dir = os.path.join(save_dir, 'in.npy')
            res_dir = os.path.join(save_dir, 'res.npy')
            conv_dir = os.path.join(save_dir, 'conv.npy')
            # fft_dir = os.path.join(save_dir, 'fft.npy')
            # out_dir = os.path.join(save_dir, 'out.npy')

            np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(res_dir, res.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(conv_dir, conv.permute(0, 2, 3, 1).cpu().detach().numpy())
            # np.save(fft_dir, fft.permute(0, 2, 3, 1).cpu().detach().numpy())
            # np.save(out_dir, out.permute(0, 2, 3, 1).cpu().detach().numpy())
            return res
        else:
            res = self.main(x) + self.main_fft(x)
            # res += x
            return res
class ResFourier_complex_onlyrelu2_phase2(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_onlyrelu2_phase2, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = fft_bench_complex_mlp_onlyrelu_clamp(n_feat, norm, window_size=None, act_max=0.)

    def forward(self, x, save_dir=None):

        res = self.main(x) + self.main_fft(x)
        # res += x
        return res
class ResFourier_complex_onlyrelu2_high8(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_onlyrelu2_high8, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = act_high_freq_r(n_feat, norm)

    def forward(self, x, save_dir=None):
        res = self.main(x) + self.main_fft(x)
        # res += x
        return res
class ResFourier_complex_onlyrelu2_low8(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_onlyrelu2_low8, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = act_low_freq_r(n_feat, norm)

    def forward(self, x, save_dir=None):
        res = self.main(x) + self.main_fft(x)
        # res += x
        return res
class ResFourier_complex_onlyrelu2_act1(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_onlyrelu2_act1, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = fft_bench_complex_mlp_onlyrelu_clamp(n_feat, norm, window_size=None, act_min=100.)

    def forward(self, x, save_dir=None):

        res = self.main(x) + self.main_fft(x)
        # res += x
        return res
class ResFourier_complex_onlyrelu2_actd1(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_onlyrelu2_actd1, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = fft_bench_complex_mlp_onlyrelu_clamp(n_feat, norm, window_size=None, act_min=-1000.)

    def forward(self, x, save_dir=None):

        res = self.main(x) + self.main_fft(x)
        # res += x
        return res
class ResFourier_complex_onlyrelu2_silu(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_onlyrelu2_silu, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = fft_bench_complex_mlp_onlyrelu_silu(n_feat, norm, window_size=None, act_min=1.)

    def forward(self, x, save_dir=None):

        res = self.main(x) + self.main_fft(x)
        # res += x
        return res
class ResFourier_complex_onlyrelu3(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_onlyrelu3, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = fft_bench_complex_mlp_onlyrelu(n_feat, norm, window_size=None)

    def forward(self, x, save_dir=None):


        res = self.main_fft(self.main(x)) + x
        # res += x
        return res
class ResFourier_complex_onlyrelu4(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_onlyrelu4, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = fft_bench_complex_mlp_onlyrelu(n_feat, norm, window_size=None)

    def forward(self, x, save_dir=None):


        res = self.main_fft(self.main(x) + x)
        # res += x
        return res
class ResFourier_complex_onlyrelu5(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_onlyrelu5, self).__init__()
        self.main = conv_frelu_bench2(n_feat)

    def forward(self, x, save_dir=None):

        res = self.main(x) + x
        # res += x
        return res
class ResFourier_conv_frelu(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_conv_frelu, self).__init__()
        self.main = conv_frelu_bench(n_feat)

    def forward(self, x, save_dir=None):
        res = self.main(x)
        # res += x
        return res + x
class ResFourier_complex_norelu(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_norelu, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = fft_bench_complex_mlp_norelu(n_feat, norm)

    def forward(self, x, save_dir=None):
        if save_dir is not None:
            conv = self.main(x)
            fft = self.main_fft(x, save_dir)
            res = conv + fft
            in_dir = os.path.join(save_dir, 'in.npy')
            res_dir = os.path.join(save_dir, 'res.npy')
            conv_dir = os.path.join(save_dir, 'conv.npy')
            # fft_dir = os.path.join(save_dir, 'fft.npy')
            out_dir = os.path.join(save_dir, 'out.npy')
            out = res + x
            np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(res_dir, res.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(conv_dir, conv.permute(0, 2, 3, 1).cpu().detach().numpy())
            # np.save(fft_dir, fft.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(out_dir, out.permute(0, 2, 3, 1).cpu().detach().numpy())
            return out
        else:
            res = self.main(x) + self.main_fft(x)
            # res += x
            return res + x
class ResFourier_complex_norelu2(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_norelu2, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = fft_bench_complex_mlp_norelu2(n_feat, norm)

    def forward(self, x, save_dir=None):

        res = self.main(x) + self.main_fft(x)
        # res += x
        return res + x
class ResFourier_complex_high_low_freq(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_high_low_freq, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = fft_bench_complex_mlp_high_low_freq(n_feat, norm)

    def forward(self, x, save_dir=None):
        if save_dir is not None:
            conv = self.main(x)
            fft = self.main_fft(x, save_dir)
            res = conv + fft
            in_dir = os.path.join(save_dir, 'in.npy')
            res_dir = os.path.join(save_dir, 'res.npy')
            conv_dir = os.path.join(save_dir, 'conv.npy')
            # fft_dir = os.path.join(save_dir, 'fft.npy')
            out_dir = os.path.join(save_dir, 'out.npy')
            out = res + x
            np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(res_dir, res.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(conv_dir, conv.permute(0, 2, 3, 1).cpu().detach().numpy())
            # np.save(fft_dir, fft.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(out_dir, out.permute(0, 2, 3, 1).cpu().detach().numpy())
            return out
        else:
            res = self.main(x) + self.main_fft(x)
            # res += x
            return res + x
class ResFourier_complex_low_freq(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_low_freq, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = fft_bench_complex_mlp_low_freq(n_feat, norm)

    def forward(self, x, save_dir=None):
        if save_dir is not None:
            conv = self.main(x)
            fft = self.main_fft(x, save_dir)
            res = conv + fft
            in_dir = os.path.join(save_dir, 'in.npy')
            res_dir = os.path.join(save_dir, 'res.npy')
            conv_dir = os.path.join(save_dir, 'conv.npy')
            # fft_dir = os.path.join(save_dir, 'fft.npy')
            out_dir = os.path.join(save_dir, 'out.npy')
            out = res + x
            np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(res_dir, res.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(conv_dir, conv.permute(0, 2, 3, 1).cpu().detach().numpy())
            # np.save(fft_dir, fft.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(out_dir, out.permute(0, 2, 3, 1).cpu().detach().numpy())
            return out
        else:
            res = self.main(x) + self.main_fft(x)
            # res += x
            return res + x
class ResFourier_complex_high_freq(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_high_freq, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = fft_bench_complex_mlp_high_freq(n_feat, norm)

    def forward(self, x, save_dir=None):
        if save_dir is not None:
            conv = self.main(x)
            fft = self.main_fft(x, save_dir)
            res = conv + fft
            in_dir = os.path.join(save_dir, 'in.npy')
            res_dir = os.path.join(save_dir, 'res.npy')
            conv_dir = os.path.join(save_dir, 'conv.npy')
            # fft_dir = os.path.join(save_dir, 'fft.npy')
            out_dir = os.path.join(save_dir, 'out.npy')
            out = res + x
            np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(res_dir, res.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(conv_dir, conv.permute(0, 2, 3, 1).cpu().detach().numpy())
            # np.save(fft_dir, fft.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(out_dir, out.permute(0, 2, 3, 1).cpu().detach().numpy())
            return out
        else:
            res = self.main(x) + self.main_fft(x)
            # res += x
            return res + x
class ResFourier_complex_low_freq_r(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_low_freq_r, self).__init__()
        self.main = conv_bench(n_feat)

        self.main_fft = fft_bench_complex_mlp_low_freq_r(n_feat, norm, win=256, r=256//8)

    def forward(self, x, save_dir=None):
        if save_dir is not None:
            conv = self.main(x)
            fft = self.main_fft(x, save_dir)
            res = conv + fft
            in_dir = os.path.join(save_dir, 'in.npy')
            res_dir = os.path.join(save_dir, 'res.npy')
            conv_dir = os.path.join(save_dir, 'conv.npy')
            # fft_dir = os.path.join(save_dir, 'fft.npy')
            out_dir = os.path.join(save_dir, 'out.npy')
            out = res + x
            np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(res_dir, res.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(conv_dir, conv.permute(0, 2, 3, 1).cpu().detach().numpy())
            # np.save(fft_dir, fft.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(out_dir, out.permute(0, 2, 3, 1).cpu().detach().numpy())
            return out
        else:
            res = self.main(x) + self.main_fft(x)
            # res += x
            return res + x
class ResFourier_complex_high_freq_r(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_high_freq_r, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = fft_bench_complex_mlp_high_freq_r(n_feat, norm, win=256, r=256//16)

    def forward(self, x, save_dir=None):
        if save_dir is not None:
            conv = self.main(x)
            fft = self.main_fft(x, save_dir)
            res = conv + fft
            in_dir = os.path.join(save_dir, 'in.npy')
            res_dir = os.path.join(save_dir, 'res.npy')
            conv_dir = os.path.join(save_dir, 'conv.npy')
            # fft_dir = os.path.join(save_dir, 'fft.npy')
            out_dir = os.path.join(save_dir, 'out.npy')
            out = res + x
            np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(res_dir, res.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(conv_dir, conv.permute(0, 2, 3, 1).cpu().detach().numpy())
            # np.save(fft_dir, fft.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(out_dir, out.permute(0, 2, 3, 1).cpu().detach().numpy())
            return out
        else:
            res = self.main(x) + self.main_fft(x)
            # res += x
            return res + x
class ResFourier_complex_only_low_freq(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_only_low_freq, self).__init__()
        self.main_fft = fft_bench_complex_mlp_low_freq(n_feat, norm)

    def forward(self, x, save_dir=None):
        if save_dir is not None:

            res = self.main_fft(x, save_dir)
            in_dir = os.path.join(save_dir, 'in.npy')

            out_dir = os.path.join(save_dir, 'out.npy')
            out = res + x
            np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(out_dir, out.permute(0, 2, 3, 1).cpu().detach().numpy())
            return out
        else:
            res = self.main_fft(x)
            # res += x
            return res + x
class ResFourier_complex_only_high_freq(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_only_high_freq, self).__init__()

        self.main_fft = fft_bench_complex_mlp_high_freq(n_feat, norm)

    def forward(self, x, save_dir=None):
        if save_dir is not None:
            res = self.main_fft(x)

            in_dir = os.path.join(save_dir, 'in.npy')

            out_dir = os.path.join(save_dir, 'out.npy')
            out = res + x
            np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(out_dir, out.permute(0, 2, 3, 1).cpu().detach().numpy())
            return out
        else:
            res = self.main_fft(x)
            # res += x
            return res + x
class ResFourier_complex_1x1(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_complex_1x1, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = fft_bench_complex_mlp_1x1(n_feat, norm)

    def forward(self, x, save_dir=None):
        if save_dir is not None:
            conv = self.main(x)
            fft = self.main_fft(x)
            res = conv + fft
            in_dir = os.path.join(save_dir, 'in.npy')
            res_dir = os.path.join(save_dir, 'res.npy')
            conv_dir = os.path.join(save_dir, 'conv.npy')
            fft_dir = os.path.join(save_dir, 'fft.npy')
            out_dir = os.path.join(save_dir, 'out.npy')
            out = res + x
            np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(res_dir, res.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(conv_dir, conv.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(fft_dir, fft.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(out_dir, out.permute(0, 2, 3, 1).cpu().detach().numpy())
            return out
        else:
            res = self.main(x) + self.main_fft(x)
            # res += x
            return res + x
class ResFourier_real(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_real, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = fft_bench_real_mlp(n_feat, norm, return_2=True)

    def forward(self, x, save_dir=None):
        if save_dir is not None:
            conv = self.main(x)
            fft = self.main_fft(x)
            res = conv + fft[-1]
            in_dir = os.path.join(save_dir, 'in.npy')
            res_dir = os.path.join(save_dir, 'res.npy')
            conv_dir = os.path.join(save_dir, 'conv.npy')
            fft0_dir = os.path.join(save_dir, 'fft0.npy')
            fft1_dir = os.path.join(save_dir, 'fft_relu.npy')
            fft2_dir = os.path.join(save_dir, 'fft2.npy')
            out_dir = os.path.join(save_dir, 'out.npy')
            out = res + x
            np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(res_dir, res.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(conv_dir, conv.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(fft0_dir, fft[0].permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(fft1_dir, fft[1].permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(fft2_dir, fft[2].permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(out_dir, out.permute(0, 2, 3, 1).cpu().detach().numpy())
            return out
        else:
            res = self.main(x) + self.main_fft(x)
            # res += x
            return res + x
class ResFourier_concat(nn.Module):
    def __init__(self, n_feat, kernel_size=3, norm='backward'): # 'backward'
        super(ResFourier_concat, self).__init__()
        self.main = conv_bench(n_feat, kernel_size)
        self.main_fft = fft_bench(n_feat, norm, return_2=False)

    def forward(self, x, save_dir=None):
        if save_dir is not None:
            conv = self.main(x)
            fft = self.main_fft(x)
            res = conv + fft[-1]
            in_dir = os.path.join(save_dir, 'in.npy')
            res_dir = os.path.join(save_dir, 'res.npy')
            conv_dir = os.path.join(save_dir, 'conv.npy')
            fft0_dir = os.path.join(save_dir, 'fft0.npy')
            fft1_dir = os.path.join(save_dir, 'fft_relu.npy')
            fft2_dir = os.path.join(save_dir, 'fft2.npy')
            out_dir = os.path.join(save_dir, 'out.npy')
            out = res + x
            np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(res_dir, res.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(conv_dir, conv.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(fft0_dir, fft[0].permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(fft1_dir, fft[1].permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(fft2_dir, fft[2].permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(out_dir, out.permute(0, 2, 3, 1).cpu().detach().numpy())
            return out
        else:
            res = self.main(x) + self.main_fft(x)
            # res += x
            return res + x
class ResFourier_concat3(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResFourier_concat3, self).__init__()
        self.main = conv_bench(n_feat)
        self.main_fft = fft_bench3(n_feat, norm, return_2=False)

    def forward(self, x, save_dir=None):
        if save_dir is not None:
            conv = self.main(x)
            fft = self.main_fft(x)
            res = conv + fft[-1]
            in_dir = os.path.join(save_dir, 'in.npy')
            res_dir = os.path.join(save_dir, 'res.npy')
            conv_dir = os.path.join(save_dir, 'conv.npy')
            fft0_dir = os.path.join(save_dir, 'fft0.npy')
            fft1_dir = os.path.join(save_dir, 'fft_relu.npy')
            fft2_dir = os.path.join(save_dir, 'fft2.npy')
            out_dir = os.path.join(save_dir, 'out.npy')
            out = res + x
            np.save(in_dir, x.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(res_dir, res.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(conv_dir, conv.permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(fft0_dir, fft[0].permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(fft1_dir, fft[1].permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(fft2_dir, fft[2].permute(0, 2, 3, 1).cpu().detach().numpy())
            np.save(out_dir, out.permute(0, 2, 3, 1).cpu().detach().numpy())
            return out
        else:
            res = self.main(x) + self.main_fft(x)
            # res += x
            return res + x
class simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-10):
        super(simam_module, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
class conv_bench(nn.Module):
    def __init__(self, n_feat, kernel_size=3, act_method=nn.ReLU):
        super(conv_bench, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.act = act_method(inplace=True)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))

class conv_frelu_bench(nn.Module):
    def __init__(self, n_feat, act_method=nn.ReLU):
        super(conv_frelu_bench, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = fft_bench_complex_mlp_onlyrelu(n_feat)

    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))
class conv_frelu_bench2(nn.Module):
    def __init__(self, n_feat, act_method=nn.ReLU):
        super(conv_frelu_bench2, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = act_method(inplace=True)
        self.actf = fft_bench_complex_mlp_onlyrelu(n_feat)

    def forward(self, x):
        return self.conv2(self.actf(self.act(self.conv1(x))))
# class conv_fftconv_bench(nn.Module):
#     def __init__(self, n_feat):
#         super(conv_fftconv_bench, self).__init__()
#         kernal_size = 3
#         self.conv1 = FFTConv2d(n_feat, n_feat, kernel_size=kernal_size, stride=1, padding=kernal_size//2)
#         self.conv2 = FFTConv2d(n_feat, n_feat, kernel_size=kernal_size, stride=1, padding=kernal_size//2)
#         self.act = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.act(x)
#         x = self.conv2(x)
#         return x
# class fft_cbench(nn.Module):
#     def __init__(self, out_channel, norm='backward', act_method=nn.ReLU):
#         super(fft_cbench, self).__init__()
#         self.act_fft = act_method()
#         self.complex_weight1_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
#         self.complex_weight1_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
#         self.complex_weight2_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
#         self.complex_weight2_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
#         init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
#         init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
#         init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
#         init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
#         self.norm = norm
#     def forward(self, x):
#         _, H, W, _ = x.shape
#
#         y = torch.fft.rfft2(x, dim=(-3, -2), norm=self.norm)
#         weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
#         weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
#         y = y.permute(0, 2, 3, 1)
#         dim = -1
#         y = y @ weight1
#         y_f = torch.cat([y.real, y.imag], dim=dim)
#         y = self.act_fft(y_f)
#         y_real, y_imag = torch.chunk(y, 2, dim=dim)
#         y = torch.complex(y_real, y_imag)
#         y = y @ weight2
#         y = y.permute(0, 3, 1, 2)
#         y = torch.fft.irfft2(y, dim=(-3, -2), s=(H, W), norm=self.norm)
#         return y
class cbench(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(cbench, self).__init__()
        self.act_fft = nn.ReLU()
        self.complex_weight1_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight1_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_real = nn.Parameter(torch.Tensor(out_channel, out_channel))
        self.complex_weight2_imag = nn.Parameter(torch.Tensor(out_channel, out_channel))
        init.kaiming_uniform_(self.complex_weight1_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight1_imag, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_real, a=math.sqrt(16))
        init.kaiming_uniform_(self.complex_weight2_imag, a=math.sqrt(16))
        self.norm = norm
    def forward(self, x):
        _, H, W, _ = x.shape
        dim = 1

        weight1 = torch.complex(self.complex_weight1_real, self.complex_weight1_imag)
        weight2 = torch.complex(self.complex_weight2_real, self.complex_weight2_imag)
        y = x @ weight1
        y_f = torch.cat([y.real, y.imag], dim=dim)
        y = self.act_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = y @ weight2
        return y
def get_ifft(x, H, W, dim=1, norm='backward'):
    y_real, y_imag = torch.chunk(x, 2, dim=dim)
    x = torch.complex(y_real, y_imag)
    x = torch.fft.irfft2(x, s=(H, W), norm=norm)
    return x
class fft_bench_x(nn.Module):
    def __init__(self, n_feat, norm='backward', act_method=nn.ReLU): # 'backward'
        super(fft_bench_x, self).__init__()

        self.main_fft1 = BasicConv(n_feat, n_feat, kernel_size=1, stride=1, relu=False)
        self.main_fft2 = BasicConv(n_feat, n_feat, kernel_size=1, stride=1, relu=False)
        self.act = act_method()
        self.norm = norm

    def forward(self, y):
        _, _, H, W = y.shape
        dim = 1
        y = torch.fft.rfft2(y, norm=self.norm)

        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.main_fft1(y)
        y = self.act(y)
        y = self.main_fft2(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class fft_bench_win(nn.Module):
    def __init__(self, n_feat, win=32, norm='backward'): # 'backward'
        super(fft_bench_win, self).__init__()
        self.win = win
        self.main_fft = self.main_fft = nn.Sequential(
            BasicConv_do(n_feat*2, n_feat*2, kernel_size=1, stride=1, relu=True),
            BasicConv_do(n_feat*2, n_feat*2, kernel_size=1, stride=1, relu=False)
        )
        # self.act = act_method()
        self.norm = norm

    def forward(self, y):
        _, _, H, W = y.shape
        y, batch_list = window_partitionx(y, self.win)
        dim = 1
        y = torch.fft.rfft2(y, norm=self.norm)
        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.main_fft(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        y = window_reversex(y, self.win, H, W, batch_list)
        return y
class fft_bench(nn.Module):
    def __init__(self, n_feat, norm='backward', act_method=nn.ReLU, return_2=False): # 'backward'
        super(fft_bench, self).__init__()

        # self.conv1 = BasicConv(n_feat*2, n_feat*2, kernel_size=1, stride=1, relu=False)
        # self.conv2 = BasicConv(n_feat*2, n_feat*2, kernel_size=1, stride=1, relu=False)
        self.conv1 = nn.Conv2d(n_feat*2, n_feat*2, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(n_feat*2, n_feat*2, kernel_size=1, stride=1, padding=0, bias=False)
        self.act = act_method(inplace=True)
        self.norm = norm
        self.return_2 = return_2

    def forward(self, y):
        if self.return_2:
            _, _, H, W = y.shape
            dim = 1
            y = torch.fft.rfft2(y, norm=self.norm)
            y = torch.cat([y.real, y.imag], dim=dim)
            y1 = self.conv1(y)
            y2 = self.act(y1)
            y = self.conv2(y2)
            y_real, y_imag = torch.chunk(y, 2, dim=dim)
            y = torch.complex(y_real, y_imag)
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
            y1 = get_ifft(y1, H, W, dim=dim)
            y2 = get_ifft(y2, H, W, dim=dim)
            return y1, y2, y
        else:
            _, _, H, W = y.shape
            dim = 1
            y = torch.fft.rfft2(y, norm=self.norm)
            y = torch.cat([y.real, y.imag], dim=dim)
            y = self.conv2(self.act(self.conv1(y)))
            y_real, y_imag = torch.chunk(y, 2, dim=dim)
            y = torch.complex(y_real, y_imag)
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
            return y
class fft_bench3(nn.Module):
    def __init__(self, n_feat, norm='backward', act_method=nn.ReLU, return_2=False): # 'backward'
        super(fft_bench3, self).__init__()

        # self.conv1 = BasicConv(n_feat*2, n_feat*2, kernel_size=1, stride=1, relu=False)
        # self.conv2 = BasicConv(n_feat*2, n_feat*2, kernel_size=1, stride=1, relu=False)
        self.conv1 = nn.Conv2d(n_feat*2, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(n_feat*2, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = act_method(inplace=True)
        self.norm = norm
        self.return_2 = return_2

    def forward(self, y):
        if self.return_2:
            _, _, H, W = y.shape
            dim = 1
            y = torch.fft.rfft2(y, norm=self.norm)
            y = torch.cat([y.real, y.imag], dim=dim)
            y1 = self.conv1(y)
            y2 = self.act(y1)
            y = self.conv2(y2)
            y_real, y_imag = torch.chunk(y, 2, dim=dim)
            y = torch.complex(y_real, y_imag)
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
            y1 = get_ifft(y1, H, W, dim=dim)
            y2 = get_ifft(y2, H, W, dim=dim)
            return y1, y2, y
        else:
            _, _, H, W = y.shape
            dim = 1
            y = torch.fft.rfft2(y, norm=self.norm)
            y = torch.cat([y.real, y.imag], dim=dim)
            y = self.conv2(self.act(self.conv1(y)))
            y_real, y_imag = torch.chunk(y, 2, dim=dim)
            y = torch.complex(y_real, y_imag)
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
            return y
class fft_bench_coord(nn.Module):
    def __init__(self, n_feat, norm='backward', act_method=nn.ReLU): # 'backward'
        super(fft_bench_coord, self).__init__()

        self.main_fft1 = BasicConv(n_feat * 2 + 2, n_feat * 2, kernel_size=1, stride=1, relu=False)
        self.main_fft2 = BasicConv(n_feat * 2 + 2, n_feat * 2, kernel_size=1, stride=1, relu=False)
        self.act = act_method()
        self.norm = norm

    def forward(self, y):
        _, _, H, W = y.shape
        dim = 1
        y = torch.fft.rfft2(y, norm=self.norm)

        y = torch.cat([y.real, y.imag], dim=dim)
        y = get_Coord(y)
        y = self.main_fft1(y)
        y = self.act(y)
        y = get_Coord(y)
        y = self.main_fft2(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class fft_target_bench(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(fft_target_bench, self).__init__()

        self.main_fft1 = BasicConv(n_feat * 2, n_feat * 2, kernel_size=1, stride=1, relu=False)
        self.main_fft2 = BasicConv(n_feat * 2, n_feat * 2, kernel_size=1, stride=1, relu=False)
        self.act = nn.ReLU()
        self.norm = norm

    def forward(self, y):
        _, _, H, W = y.shape
        dim = 1
        y = torch.fft.rfft2(y, s=[720, 1280], norm=self.norm)

        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.main_fft1(y)
        y = self.act(y)
        y = self.main_fft2(y)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class fft_bench_phase(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(fft_bench_phase, self).__init__()

        self.main = nn.Sequential(
            BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=True),
            BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=False)
        )
        self.main_phase = nn.Sequential(
            BasicConv(n_feat, n_feat, kernel_size=1, stride=1, relu=True),
            BasicConv(n_feat, n_feat, kernel_size=1, stride=1, relu=False)
        )
        # self.act = nn.ReLU()
        self.norm = norm
        # self.norm_layer = nn.InstanceNorm2d(n_feat)
    def forward(self, y):
        _, _, H, W = y.shape
        # dim = 1
        y = torch.fft.rfft2(y, norm=self.norm)
        _, _, h, w = y.shape
        A = torch.abs(y) # / (H * W)

        p = pi.to(y.device)
        phase = torch.angle(y) / p
        # print(phase.max(), phase.min())
        phase = self.main_phase(phase)
        phase = torch.clamp(phase, -1., 1.)
        phase = phase * p

        A_mean = torch.mean(A, dim=[-2, -1], keepdim=True)
        A_std = torch.var(A, dim=[-2, -1], keepdim=True)
        A = (A - A_mean) / (A_std + 1e-8)
        A = self.main(A)
        # A = rearrange(A, 'b c h w -> b c (h w)')
        # A = torch.softmax(A, dim=-1)
        # A = rearrange(A, 'b c (h w) -> b c h w', h=h, w=w)
        # print(A.dtype, phase.dtype)
        y = torch.complex(A * torch.cos(phase), A * torch.sin(phase))
        # y = torch.complex(torch.cos(phase), torch.sin(phase))
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y # torch.tanh(y) # torch.tanh(self.norm_layer(y))
class div_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-10):
        super(div_module, self).__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h
        x = x / n
        # x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda))
        return x

class fft_bench_div(nn.Module):
    def __init__(self, n_feat, norm='backward'):  # 'backward'
        super(fft_bench_div, self).__init__()

        self.main_fft1 = BasicConv(n_feat * 2, n_feat * 2, kernel_size=1, stride=1, relu=False)
        self.main_fft2 = BasicConv(n_feat * 2, n_feat * 2, kernel_size=1, stride=1, relu=False)
        self.act = nn.ReLU()
        self.div_m = div_module()
        self.norm = norm

    def forward(self, y):
        _, _, H, W = y.shape

        dim = 1
        y = torch.fft.rfft2(y, norm=self.norm)
        y = torch.cat([y.real, y.imag], dim=dim)
        y = self.div_m(y)

        y = self.main_fft1(y)
        y = self.act(y)
        y = self.main_fft2(y)
        # y = torch.tanh(y)

        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class ResBlock_fft_bench_windowShuffle(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResBlock_fft_bench_windowShuffle, self).__init__()
        self.main = nn.Sequential(
            BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=True),
            BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft = nn.Sequential(
            BasicConv(n_feat*2, n_feat*2, kernel_size=1, stride=1, relu=True),
            BasicConv(n_feat*2, n_feat*2, kernel_size=1, stride=1, relu=False)
        )
        self.dim = n_feat
        self.norm = norm
        self.shuffle = ShuffleBlock(groups=4)
    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        # y = torch.fft.rfft2(x, norm='ortho')
        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.shuffle(y)
        y_f = torch.cat([y.real, y.imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm='ortho')
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y
class ResBlock_Shuffle(nn.Module):
    def __init__(self, n_feat, norm='backward'): # 'backward'
        super(ResBlock_Shuffle, self).__init__()
        self.main1 = ResBlock_fft_bench(n_feat)
        self.main2 = ResBlock_fft_bench_windowShuffle(n_feat)
        # self.shuffle = ShuffleBlock(groups=4)

    def forward(self, x):
        x = self.main1(x)
        # x = self.shuffle(x)
        x = self.main2(x)
        return x

class ResBlock_do_fft_bench_eval(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_do_fft_bench_eval, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft = nn.Sequential(
            BasicConv_do_eval(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=True),
            BasicConv_do_eval(out_channel*2, out_channel*2, kernel_size=1, stride=1, relu=False)
        )
        self.dim = out_channel
        self.norm = norm
    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        # y = torch.fft.rfft2(x, norm='ortho')
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm='ortho')
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y


class InsNorm(nn.Module):
    def __init__(self, ):
        super(InsNorm, self).__init__()

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


if __name__=='__main__':
    import torch
    win = 128
    Method = DCT_test(win).cuda()
    x = torch.randn(1,1,win,win)
    x = x.cuda()
    x_idct, x_dct = Method(x)

    print(x_dct.squeeze())
    print(x_dct.max(), x_dct.min())
    print(torch.allclose(x_idct, x))
    print(torch.max(x_idct-x))
    # print(x.shape, y.shape)