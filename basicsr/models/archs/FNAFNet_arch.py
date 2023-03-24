# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import * # LayerNorm2d, window_reversex, window_partitionx, FFT_ReLU, Attention_win
from basicsr.models.archs.local_arch import Local_Base
from basicsr.models.archs.up_down import *
# from basicsr.models.archs.natten import NeighborhoodAttention

class SimpleGate(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class WNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.window_size = window_size
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
        x = x * self.sca(x)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x = window_reversex(x, self.window_size, H, W, batch_list)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class WNAFBlock_ffc3_2block(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.window_size = window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        if window_size_fft >= 0:
            self.fft_block1 = fft_bench_complex_mlp(c, DW_Expand, window_size=window_size_fft, bias=True) # , act_method=nn.GELU
            self.fft_block2 = fft_bench_complex_mlp(c, DW_Expand, window_size=window_size_fft, bias=True)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()


    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
            # x_, _ = window_partitionx(x_, self.window_size)
        # x = x + self.fft_block(x_)
        x = x * self.sca(x)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x = window_reversex(x, self.window_size, H, W, batch_list)
        x = self.conv3(x)
        if self.window_size_fft >= 0:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        if self.window_size_fft >= 0:
            x = self.norm2(y)
            x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        else:
            x = self.conv4(self.norm2(y))
            x = self.sg(x)
            x = self.conv5(x)

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class WNAFBlock_ffc3_sin_2block(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.window_size = window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        if window_size_fft >= 0:
            self.fft_block1 = fft_bench_complex_mlp(c, DW_Expand, window_size=window_size_fft, bias=True) # , act_method=nn.GELU
            self.fft_block2 = fft_bench_complex_mlp(c, DW_Expand, window_size=window_size_fft, bias=True)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()


    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
            # x_, _ = window_partitionx(x_, self.window_size)
        # x = x + self.fft_block(x_)
        x = x * self.sca(x)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x = window_reversex(x, self.window_size, H, W, batch_list)
        x = self.conv3(x)
        if self.window_size_fft >= 0:
            x = x + torch.sin(self.fft_block1(x_))
        x = self.dropout1(x)
        y = inp + x * self.beta

        if self.window_size_fft >= 0:
            x = self.norm2(y)
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        else:
            x = self.conv4(self.norm2(y))
            x = self.sg(x)
            x = self.conv5(x)

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class WNAFBlock_ffc3_gelu_sin_2block(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=True):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        if window_size_fft is None or window_size_fft >= 0:
            self.fft_block1 = fft_bench_complex_mlp(c, DW_Expand, window_size=window_size_fft, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
            self.fft_block2 = fft_bench_complex_mlp(c, DW_Expand, window_size=window_size_fft, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()


    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
            # x_, _ = window_partitionx(x_, self.window_size)
        # x = x + self.fft_block(x_)
        x = x * self.sca(x)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x = window_reversex(x, self.window_size, H, W, batch_list)
        x = self.conv3(x)
        if self.window_size_fft is None or self.window_size_fft >= 0:
            if self.sin:
                x = x + torch.sin(self.fft_block1(x_))
            else:
                x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        if self.window_size_fft is None or self.window_size_fft >= 0:
            x = self.norm2(y)
            if self.sin:
                x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
            else:
                x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))
        else:
            x = self.conv4(self.norm2(y))
            x = self.sg(x)
            x = self.conv5(x)

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class WNAFBlock_ffc3_gelu_sin_2block_flops(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.window_size = window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        if window_size_fft >= 0:
            self.fft_block1 = fft_bench_complex_mlp_flops(c, DW_Expand, window_size=window_size_fft, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
            self.fft_block2 = fft_bench_complex_mlp_flops(c, DW_Expand, window_size=window_size_fft, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()


    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x, batch_list = window_partitionx(x, self.window_size)
            # x_, _ = window_partitionx(x_, self.window_size)
        # x = x + self.fft_block(x_)
        x = x * self.sca(x)
        if self.window_size is not None and (H != self.window_size or W != self.window_size):
            x = window_reversex(x, self.window_size, H, W, batch_list)
        x = self.conv3(x)
        if self.window_size_fft >= 0:
            x = x + torch.sin(self.fft_block1(x_))
        x = self.dropout1(x)
        y = inp + x * self.beta

        if self.window_size_fft >= 0:
            x = self.norm2(y)
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        else:
            x = self.conv4(self.norm2(y))
            x = self.sg(x)
            x = self.conv5(x)

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x

class FNAFNet(nn.Module):

    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1],
                 window_size_e=[64,32,16,8], window_size_m=[8], window_size_e_fft=[64, 32, 16, -1], window_size_m_fft=[-1],
                 window_sizex_e=[8,8,8,8], window_sizex_m=[8], num_heads_e=[1, 2, 4, 8], num_heads_m=[16]):
        super().__init__()
        # num_heads_e = [1, 2, 4, 8]
        # num_heads_m = [16]
        # num_heads_e = [4, 4, 4, 4]  # [32, 64, 128, 256]
        # num_heads_m = [4]
        num_heads_d = num_heads_e[::-1] # [8, 4, 2, 1]

        print(num_heads_e, window_size_e, window_size_e_fft, window_sizex_e)

        window_size_d = window_size_e[::-1]
        window_size_d_fft = window_size_e_fft[::-1]
        window_sizex_d = window_sizex_e[::-1]

        NAFBlock = WNAFBlock_ffc3_gelu_sin_2block

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        # self.fft_head = FFT_head()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for i in range(len(enc_blk_nums)):
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, num_heads_e[i], window_size_e[i], window_size_e_fft[i], window_sizex_e[i]) for _ in range(enc_blk_nums[i])]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
                # DownModule(chan, 2*chan)
            )
            chan = chan * 2
        # print(NAFBlock)
        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan, num_heads_m[0], window_size_m[0], window_size_m_fft[0], window_sizex_m[0]) for _ in range(middle_blk_num)]
            )

        for j in range(len(dec_blk_nums)):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                    # UpModule(chan, chan//2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, num_heads_d[j], window_size_d[j], window_size_d_fft[j], window_sizex_d[j]) for _ in range(dec_blk_nums[j])]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        loss = 0.
        B, C, H, W = inp.shape

        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):

            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        # x = self.fft_head(x)
        output = x + inp
        # output = {'img': x[:, :, :H, :W], 'loss': loss}
        return output[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class FNAFNetLocal(Local_Base, FNAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        FNAFNet.__init__(self, *args, **kwargs)
        # train_size = (1, 3, 64, 64)
        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))
        # base_size = (int(H), int(W))
        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

if __name__ == '__main__':
    import resource
    def using(point=""):
        # print(f'using .. {point}')
        usage = resource.getrusage(resource.RUSAGE_SELF)
        global Total, LastMem

        # if usage[2]/1024.0 - LastMem > 0.01:
        # print(point, usage[2]/1024.0)
        print(point, usage[2] / 1024.0)

        LastMem = usage[2] / 1024.0
        return usage[2] / 1024.0

    img_channel = 3
    width = 32
    
    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    
    print('enc blks', enc_blks, 'middle blk num', middle_blk_num, 'dec blks', dec_blks, 'width' , width)
    
    using('start . ')
    net = WNAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    # using('network .. ')
    #
    # # for n, p in net.named_parameters()
    # #     print(n, p.shape)
    #
    inp = torch.randn((1, 3, 128, 128))
    print(inp.shape, inp.max())
    net = net.cuda()
    inp = inp.cuda()
    out = net(inp)
    print(torch.mean(out-inp))
    # final_mem = using('end .. ')
    # out.sum().backward()

    # out.sum().backward()

    # using('backward .. ')

    # exit(0)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print('FLOPs: ', macs)
    print('params: ', params)

    # print('total .. ', params * 8 + final_mem)



