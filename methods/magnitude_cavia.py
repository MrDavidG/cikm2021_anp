import sys

sys.path.append('..')

from components.backbones import get_backbone
from components.datasets import get_dataset
from components.optimizer import Adam
from components.utils import test_masks
from components.utils import get_cr
from components.learner_cavia import Learner
from torch.utils.data import DataLoader

import torch.nn.functional as F
import numpy as np

import argparse
import torch
import os

SECOND = True

PATH_BASE = None

PATH_DATA = None
N_CLS = None
DATA = None
DATA_SINGLE = None

NAME_DATA = None
NAME_BACKBONE = None

SECOND = True


def magnitude(args, config, path_load, device):
    print('Load net from', path_load)
    save_dict = torch.load(path_load)
    state_dict = save_dict['state_dict']
    target = save_dict['target']

    net = Learner(config, args.num_context_params, args.context_in).to(device)
    net.load_state_dict(state_dict)

    # Load data
    data = DATA_SINGLE(PATH_DATA, args.imgsz, 'train', target)
    db = DataLoader(data, batch_size=24, shuffle=True, num_workers=8, pin_memory=True)
    data_test = DATA_SINGLE(PATH_DATA, args.imgsz, 'test', target)
    db_test = DataLoader(data_test, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    if 'masks_dict' in save_dict.keys():
        masks_dict = save_dict['masks_dict']
    else:
        masks_dict = dict()

    # prune
    for _, alpha in enumerate(args.cr_steps):
        weights_map = net.weights_map
        for name in net.layers_name:
            name_w, name_b = weights_map[name]

            w = state_dict[name_w]
            b = state_dict[name_b]

            w_reshape_abs = torch.abs(torch.reshape(w, [-1]))

            index_w = torch.sum(w_reshape_abs == 0) + int(w_reshape_abs.size(0) * alpha)
            threshold_w = torch.sort(w_reshape_abs)[0][index_w]

            mask_w = (torch.abs(w) >= threshold_w).cpu().numpy().astype(np.int32)
            state_dict[name_w] = w * torch.tensor(mask_w, dtype=torch.float32, requires_grad=True).to(device)
            masks_dict[name_w] = mask_w

            b_abs = torch.abs(b)

            index_b = torch.sum(b_abs == 0) + int(b.size(0) * alpha)
            threshold_b = torch.sort(b_abs)[0][index_b]

            mask_b = (b_abs >= threshold_b).cpu().numpy().astype(np.int32)
            state_dict[name_b] = b * torch.tensor(mask_b, dtype=torch.float32, requires_grad=True).to(device)
            masks_dict[name_b] = mask_b

        net.load_state_dict(state_dict)

        test_masks(masks_dict, net.state_dict())
        cr = get_cr(net, masks_dict)

        # Retrain
        for epoch in range(args.epoch_retrain):
            opt = Adam(net.parameters(), masks_dict, lr=0.001 * 0.1 ** (epoch // 5))
            for step, (x, y) in enumerate(db):
                net.train()
                x, y = x.to(device), y.to(device)

                logits = net(x)
                loss = F.cross_entropy(logits, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

                if (step + 1) % 10 == 0:
                    net.eval()
                    correct = 0.
                    for x, y in db_test:
                        x, y = x.to(device), y.to(device)

                        logits_q = net(x)

                        with torch.no_grad():
                            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                            correct += torch.eq(pred_q, y).sum().item()

                    acc = correct / len(db_test.dataset)
                    print('\rEpoch', epoch, 'step', step + 1, 'loss', '%.4f' % loss.cpu().detach().numpy(), 'Test',
                          step, 'acc', '%.4f' % acc, end='')
        print('')

        # Save
        path_save = '/'.join(path_load.split('/')[:-1])
        if not os.path.exists(path_save):
            os.makedirs(path_save)

        path_model = '%s/net-cr_%.2f.pkl' % (path_save, cr)
        print('Save as %s' % path_model)
        torch.save(
            {
                'state_dict': net.state_dict(),
                'masks_dict': masks_dict,
                'target': target,
                'cr': cr,
                'config': config
            },
            path_model
        )


def exp(n_way,
        k_spt,
        dataset,
        backbone,
        task,
        path_model=None,
        k_qry=15,
        cr_steps=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05],
        epoch_test=5,
        epoch_retrain=10,
        epoch=60000,
        num_context_params=100,
        context_in=[False, False, True, False, False],
        imgsz=84,
        imgc=3,
        task_num=4,
        meta_lr=1e-3,
        update_lr=0.01,
        update_step=5,
        update_step_test=10,
        log_print=False):
    argparser = argparse.ArgumentParser()
    # prune
    argparser.add_argument('--cr_steps', type=list, help='cr at each iteration', default=cr_steps)
    argparser.add_argument('--epoch_test', type=int, help='number of test times', default=epoch_test)
    argparser.add_argument('--epoch_retrain', type=int, help='number of retraining', default=epoch_retrain)
    # maml
    argparser.add_argument('--num_context_params', type=int, help='number of context parameter vector',
                           default=num_context_params)
    argparser.add_argument('--context_in', type=list, help='whether add context in each layer', default=context_in)
    argparser.add_argument('--log_print', type=bool, help='print logs for meta training', default=log_print)
    argparser.add_argument('--epoch', type=int, help='total number of task sets', default=epoch)
    argparser.add_argument('--n_way', type=int, help='n way', default=n_way)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=k_spt)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=k_qry)
    argparser.add_argument('--imgsz', type=int, help='image size', default=imgsz)
    argparser.add_argument('--imgc', type=int, help='number of channels for images', default=imgc)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=task_num)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=meta_lr)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=update_lr)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=update_step)
    argparser.add_argument('--update_step_test', type=int, help='update steps for fine tunning',
                           default=update_step_test)
    args = argparser.parse_args()

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    device = torch.device('cuda')

    # 获得config
    config = get_backbone(backbone, n_way, imgsz, imgsz)

    # 获得dataset
    global DATA, DATA_SINGLE, PATH_DATA, N_CLS, PATH_BASE, NAME_DATA, NAME_BACKBONE
    DATA, DATA_SINGLE, PATH_DATA, N_CLS, PATH_BASE = get_dataset(dataset)
    NAME_DATA = dataset
    NAME_BACKBONE = backbone

    if task == 'magnitude':
        magnitude(args, config, path_model, device)
    else:
        raise RuntimeError
