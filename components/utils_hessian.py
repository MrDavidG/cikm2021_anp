import torch.nn.functional as F
import numpy as np

import torch
import math


def extract_image_patches(x, kernel, stride=1, dilation=1):
    # Do TF 'SAME' Padding
    b, c, h, w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = F.pad(x, (pad_row // 2, pad_row - pad_row // 2, pad_col // 2, pad_col - pad_col // 2))

    # Extract patches
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches = patches.permute(0, 4, 5, 1, 2, 3).contiguous()

    return patches.view(b, -1, patches.shape[-2], patches.shape[-1])


def generate_conv_hessian(layer_input, layer_kernel, layer_stride, device):
    """

    :param layer_input: [n,c,h,w]
    :param layer_kernel:
    :param layer_stride:
    :return:
    """
    # [32,27,84,84]
    patches = extract_image_patches(layer_input, kernel=layer_kernel, stride=layer_stride).to(device)

    # [32,28,84,84]->[32,84,84,28]
    vect_w_b = torch.cat([patches, torch.ones(patches.size(0), 1, patches.size(2), patches.size(3)).to(device)],
                         dim=1).permute(0, 2, 3, 1)

    a = torch.unsqueeze(vect_w_b, dim=-1)
    b = torch.unsqueeze(vect_w_b, dim=3)
    # print(a.size(), b.size())
    outprod = torch.matmul(a, b)

    hessian = torch.mean(outprod, dim=[0, 1, 2])
    return hessian.detach().cpu().numpy()


def generate_fc_hessian(layer_input, device):
    # [32, 800, 1]
    a = torch.unsqueeze(layer_input, dim=-1).to(device)
    # [32, 801, 1]
    vect_w_b = torch.cat([a, torch.ones([a.size(0), 1, 1]).to(device)], dim=1)
    # [32, 801, 1] * [32, 1, 801]
    outprod = torch.matmul(vect_w_b, vect_w_b.permute([0, 2, 1]))
    hessian = torch.mean(outprod, dim=0)
    return hessian.detach().cpu().numpy()

