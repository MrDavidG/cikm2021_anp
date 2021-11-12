import torch
from components.backbones import get_backbone
from components.utils import get_cr
from components.learner import Learner

import numpy as np


def get_flops(config, imgc, imgsz, path_model):
    save_dict = torch.load(path_model)
    masks_dict = save_dict['masks_dict']

    learner = Learner(config, imgc, imgsz)
    get_cr(learner, masks_dict)

    input = torch.rand(size=(1, 3, 84, 84), dtype=torch.float32)
    _ = learner(input)

    weights_map = learner.weights_map
    flops, flops_cr = 0, 0
    for layer_name in learner.layers_name:
        name_w, name_b = weights_map[layer_name]
        mask_w, mask_b = masks_dict[name_w], masks_dict[name_b]
        if layer_name.startswith('c'):
            # n, c, h, w
            (n, c, h, w) = learner.layers_input[layer_name].shape

            # n_o, n_i, k, k
            (n_o, n_i, k1, k2) = mask_w.shape

            flops += 2 * k1 * k2 * n_i * n_o * h * w
            flops_cr += 2 * np.sum(mask_w) * h * w + h * w * (np.sum(mask_b) - mask_b.shape[0])
        else:
            (n_o, n_i) = mask_w.shape

            flops += 2 * n_o * n_i
            flops_cr += 2 * np.sum(mask_w) + np.sum(mask_b) - n_o
    return flops, flops_cr, flops_cr / float(flops)


if __name__ == '__main__':
    config = get_backbone('conv32_br', 5, 84, 84)
    for path_model in [
        '/local/home/david/Remote/VibNet_Pytorch/model/maml_lobs/cub_conv32_br/5-way_5-shot/net_cr-0.06_acc-0.3895.pkl',
        '/local/home/david/Remote/VibNet_Pytorch/model/maml_lobs/cub_conv32_br/5-way_5-shot/net_cr-0.11_acc-0.5066.pkl',
        '/local/home/david/Remote/VibNet_Pytorch/model/maml_lobs/cub_conv32_br/5-way_5-shot/net_cr-0.16_acc-0.4867.pkl',
        '/local/home/david/Remote/VibNet_Pytorch/model/maml_lobs/cub_conv32_br/5-way_5-shot/net_cr-0.21_acc-0.4855.pkl',
        '/local/home/david/Remote/VibNet_Pytorch/model/maml_lobs/cub_conv32_br/5-way_5-shot/net_cr-0.50_acc-0.5128.pkl',
        '/local/home/david/Remote/VibNet_Pytorch/model/maml_lobs/cub_conv32_br/5-way_5-shot/net_cr-0.70_acc-0.5228.pkl'
    ]:
        print(get_flops(config=config,
                        imgc=3,
                        imgsz=84,
                        path_model=path_model))
