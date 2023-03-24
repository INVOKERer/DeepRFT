import torch
import torch.nn.functional as F
import torch.nn as nn
def check_image_size(x, padder_size, mode='reflect'):
    _, _, h, w = x.size()
    if isinstance(padder_size, int):
        padder_size_h = padder_size
        padder_size_w = padder_size
    else:
        padder_size_h, padder_size_w = padder_size
    mod_pad_h = (padder_size_h - h % padder_size_h) % padder_size_h
    mod_pad_w = (padder_size_w - w % padder_size_w) % padder_size_w
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode=mode)
    return x

def window_partitions(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    if isinstance(window_size, int):
        window_size = [window_size, window_size]
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size[0], window_size[1])
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
    if isinstance(window_size, int):
        window_size = [window_size, window_size]
    C = windows.shape[1]
    # print('C: ', C)
    x = windows.view(-1, H // window_size[0], W // window_size[1], C, window_size[0], window_size[1])
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
    # print(windows[:batch_list[0], ...].shape)
    x_main = window_reverses(windows[:batch_list[0], ...], window_size, h, w)
    B, C, _, _ = x_main.shape
    # print('windows: ', windows.shape)
    # print('batch_list: ', batch_list)
    if torch.is_complex(windows):
        res = torch.complex(torch.zeros([B, C, H, W]), torch.zeros([B, C, H, W]))
        res = res.to(windows.device)
    else:
        res = torch.zeros([B, C, H, W], device=windows.device)

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
    x_main, b_main = window_partitionx(x[:, :, s_h:, s_w:], window_size)
    # print(x_main.shape, b_main, x[:, :, s_h:, s_w:].shape)
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
        # print(windows[:batch_list[-4], ...].shape, batch_list[:-3], H-s_h, W-s_w)
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
class WindowPartition(nn.Module):
    def __init__(self, window_size=8, shift_size=0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
    def forward(self, x):
        H, W = x.shape[-2:]
        if self.window_size is not None and (H > self.window_size and W > self.window_size):
            if not self.shift_size:
                x, batch_list = window_partitionx(x, self.window_size)
                return x, batch_list
            else:
                x, batch_list = window_partitionxy(x, self.window_size, [self.shift_size, self.shift_size])
                return x, batch_list
        else:
            return x, []

class WindowReverse(nn.Module):
    def __init__(self, window_size=8, shift_size=0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
    def forward(self, x, H, W, batch_list):
        # print(x.shape, batch_list)
        if len(batch_list) > 0 and self.window_size is not None and (H > self.window_size and W > self.window_size):
            if not self.shift_size:
                x = window_reversex(x, self.window_size, H, W, batch_list)
            else:
                x = window_reversexy(x, self.window_size, H, W, batch_list, [self.shift_size, self.shift_size])
        return x