import sys

sys.path.append('..')

from components.utils_hessian_tf import generate_hessian_inv_Woodbury_maml_fsl
from torch.nn import Module
from components.backbones import get_backbone
from components.datasets import get_dataset
from components.optimizer import Adam
from components.learner import Learner
from components.utils import test_masks
from components.utils import mask_grad
from components.utils import unfold, fold_weights, get_cr
from torch.utils.data import DataLoader
from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

import torch.nn.functional as F
import numpy as np

import argparse
import torch
import os

PATH_BASE = None

PATH_DATA = None
N_CLS = None
DATA = None
DATA_SINGLE = None

NAME_DATA = None
NAME_BACKBONE = None

SECOND = True

# gpu 0
# os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4eec6600-f5e3-f385-9b14-850ae9a2b236'
# gpu 1
os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4b0856cd-c698-63a2-0b6e-9a33d380f9c4'


def train_net(args, config, target, path_model, device):
    print('Train net for target', target)

    # Load data
    data = DATA_SINGLE(PATH_DATA, args.imgsz, 'train', target)
    db = DataLoader(data, batch_size=24, shuffle=True, num_workers=8, pin_memory=True)
    data_test = DATA_SINGLE(PATH_DATA, args.imgsz, 'test', target)
    db_test = DataLoader(data_test, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    # Load net
    save_dict = torch.load(path_model)
    state_dict = save_dict['state_dict']
    if 'masks_dict' in save_dict.keys():
        masks_dict = save_dict['masks_dict']
    else:
        masks_dict = None

    net = Learner(config, args.imgc, args.imgsz).to(device)
    net.load_state_dict(state_dict)
    # opt = Adam(net.parameters(), masks_dict, lr=0.01)

    losses = list()
    losses_q = list()
    accs = list()
    for epoch in range(50):
        for step, (x, y) in enumerate(db):
            # if step > 5:
            #     break
            net.train()
            x, y = x.to(device), y.to(device)

            logits = net(x)
            loss = F.cross_entropy(logits, y)

            net.zero_grad()
            loss.backward()
            for p in net.parameters():
                p.data.add_(-0.01, p.grad.data)
            net.zero_grad()

            # opt.zero_grad()
            # loss.backward()
            # opt.step()

            print('Epoch', epoch, 'step', step + 1, 'loss', '%.4f' % loss.cpu().detach().numpy(), end='')
            net.eval()
            correct = 0.
            for x_test, y_test in db_test:
                x_test, y_test = x_test.to(device), y_test.to(device)

                logits_q = net(x_test)

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct += torch.eq(pred_q, y_test).sum().item()
                    loss_q = F.cross_entropy(logits_q, y_test)

            acc = correct / len(db_test.dataset)
            print('\tTest', step, 'acc', '%.4f' % acc)

            losses.append(loss)
            losses_q.append(loss_q)
            accs.append(acc)
    test_masks(masks_dict, state_dict)
    return losses, losses_q, accs


def exp(n_way,
        k_train,
        k_spt,
        dataset,
        backbone,
        target,
        path_models,
        labels,
        imgsz=84,
        imgc=3,
        update_lr=0.01,
        ):
    argparser = argparse.ArgumentParser()
    # maml
    argparser.add_argument('--n_way', type=int, help='n way', default=n_way)
    argparser.add_argument('--k_train', type=int, help='n train', default=k_train)
    argparser.add_argument('--imgsz', type=int, help='image size', default=imgsz)
    argparser.add_argument('--imgc', type=int, help='number of channels for images', default=imgc)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=update_lr)
    args = argparser.parse_args()

    device = torch.device('cuda')

    # 获得config
    config = get_backbone(backbone, n_way, imgsz, imgsz)

    # 获得dataset
    global DATA, DATA_SINGLE, PATH_DATA, N_CLS, PATH_BASE, NAME_DATA, NAME_BACKBONE
    DATA, DATA_SINGLE, PATH_DATA, N_CLS, PATH_BASE = get_dataset(dataset)
    NAME_DATA = dataset
    NAME_BACKBONE = backbone

    loss_list, loss_q_list, accs_list = [], [], []
    for path_model in path_models:
        losses, losses_q, accs = train_net(args, config, target, path_model, device)
        loss_list.append(losses)
        loss_q_list.append(losses_q)
        accs_list.append(accs)

    torch.save({
        'loss': loss_list,
        'loss_q': loss_q_list,
        'accs': accs_list,
        'labels': labels,
    }, '/local/home/david/Remote/VibNet_Pytorch/exp_run/plot.pkl')

    plot(loss_list, '# Training Iterations', 'Training Loss', labels)

    plot(loss_q_list, '# Training Iterations', 'Loss', labels)

    plot(accs_list, '# Training Iterations', 'Accuracy', labels)


def plot(list_, xlabel, ylabel, labels):
    fontsize = 25
    fig = plt.figure(figsize=(8, 6))
    plt.rc('pdf', fonttype=42)
    plt.rc('ps', fonttype=42)
    plt.rc('font', family='Times New Roman')

    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for id, loss in enumerate(list_):
        plt.plot(np.arange(len(loss)), loss, label=labels[id])
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='EastOutside')
    plt.savefig('/local/home/david/Remote/VibNet_Pytorch/exp_run/%s.pdf' % ylabel)
    plt.show()


if __name__ == '__main__':
    exp(n_way=5, k_train=1, k_spt=15,
        dataset='mini-imagenet',
        backbone='conv32_br',
        target=[25, 37, 43, 57, 62],

        path_models=[
            # anp
            '/local/home/david/Remote/VibNet_Pytorch//model/maml_lobs/mini-imagenet_conv32_br/5-way_1-shot/net_cr-0.16_acc-0.4182.pkl',
            # orginal task + lobs + maml 10000
            '/local/home/david/Remote/VibNet_Pytorch/model/lobs/mini-imagenet_conv32_br/net1/net_cr-0.16_acc-0.2080.pkl',
            # orginal task + magnitude + maml 10000
            '/local/home/david/Remote/VibNet_Pytorch/model/magnitude/mini-imagenet_conv32_br/net1/net_cr-0.16_acc-0.2057.pkl',
            # orginal task + lobs + maml 80000
            '/local/home/david/Remote/VibNet_Pytorch/model/lobs/mini-imagenet_conv32_br/net1/net_cr-0.16_acc-0.2005.pkl',
            # orginal task + magnitude + maml 80000
            ' /local/home/david/Remote/VibNet_Pytorch/model/magnitude/mini-imagenet_conv32_br/net1/net_cr-0.16_acc-0.2005.pkl',

            # other task + lobs + maml 10000
            '/local/home/david/Remote/VibNet_Pytorch/model/lobs/mini-imagenet_conv32_br/net2/net_cr-0.16_acc-0.2130.pkl',
            # other task + magnitude + maml 10000
            '/local/home/david/Remote/VibNet_Pytorch/model/magnitude/mini-imagenet_conv32_br/net2/net_cr-0.16_acc-0.2132.pkl',
            # other task + lobs + maml 80000
            '/local/home/david/Remote/VibNet_Pytorch/model/lobs/mini-imagenet_conv32_br/net2/net_cr-0.16_acc-0.2137.pk',
            # other task + magnitude + maml 80000
            '/local/home/david/Remote/VibNet_Pytorch/model/magnitude/mini-imagenet_conv32_br/net2/net_cr-0.16_acc-0.3329.pkl'
        ],
        labels=['ANP',
                'L-OBS MAML(10000)',
                'Magnitude MAML(10000)',
                'L-OBS MAML(80000)',
                'Magnitude MAML(80000)',

                'Other L-OBS MAML(10000)',
                'Other Magnitude MAML(10000)',
                'Other L-OBS MAML(80000)',
                'Other Magnitude MAML(80000)',
                ]
        )

    # 15%
