import torch
import os
from collections import OrderedDict
import numpy as np
def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    # print(checkpoint)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # print(k)
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)


def load_checkpoint_compress_doconv(model, weights):
    checkpoint = torch.load(weights)
    # print(checkpoint)
    # state_dict = OrderedDict()
    # try:
    #     model.load_state_dict(checkpoint["state_dict"])
    #     state_dict = checkpoint["state_dict"]
    # except:
    old_state_dict = checkpoint["state_dict"]
    state_dict = OrderedDict()
    for k, v in old_state_dict.items():
        # print(k)
        name = k
        if k[:7] == 'module.':
            name = k[7:]  # remove `module.`
        state_dict[name] = v
    # state_dict = checkpoint["state_dict"]
    do_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[-1] == 'W' and k[:-1] + 'D' in state_dict:
            k_D = k[:-1] + 'D'
            k_D_diag = k_D + '_diag'
            W = v
            D = state_dict[k_D]
            D_diag = state_dict[k_D_diag]
            D = D + D_diag
            # W = torch.reshape(W, (out_channels, in_channels, D_mul))
            out_channels, in_channels, MN = W.shape
            M = int(np.sqrt(MN))
            DoW_shape = (out_channels, in_channels, M, M)
            DoW = torch.reshape(torch.einsum('ims,ois->oim', D, W), DoW_shape)
            do_state_dict[k] = DoW
        elif k[-1] == 'D' or k[-6:] == 'D_diag':
            continue
        elif k[-1] == 'W':
            out_channels, in_channels, MN = v.shape
            M = int(np.sqrt(MN))
            W_shape = (out_channels, in_channels, M, M)
            do_state_dict[k] = torch.reshape(v, W_shape)
        else:
            do_state_dict[k] = v
    model.load_state_dict(do_state_dict)
def load_checkpoint_hin(model, weights):
    checkpoint = torch.load(weights)
    # print(checkpoint)
    try:
        model.load_state_dict(checkpoint)
    except:
        state_dict = checkpoint
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    # for p in optimizer.param_groups: lr = p['lr']
    # return lr
