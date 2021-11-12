import sys

sys.path.append('..')

from components.utils_hessian_tf import generate_hessian_inv_Woodbury
from components.backbones import get_backbone
from components.datasets import get_dataset
from components.optimizer import Adam
from components.utils import test_masks
from components.utils import unfold, fold_weights, get_cr
from components.learner import Learner
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


def train_net(args, config, device, idx):
    target = np.sort(np.random.choice(np.arange(N_CLS), args.n_way, replace=False)).tolist()
    print('Train net for target', target)

    # Load data
    data = DATA_SINGLE(PATH_DATA, args.imgsz, 'train', target)
    db = DataLoader(data, batch_size=24, shuffle=True, num_workers=8, pin_memory=True)
    data_test = DATA_SINGLE(PATH_DATA, args.imgsz, 'test', target)
    db_test = DataLoader(data_test, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    # Load net
    net = Learner(config, args.imgc, args.imgsz).to(device)
    opt = Adam(net.parameters(), None, lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    for epoch in range(40):
        for step, (x, y) in enumerate(db):
            net.train()
            x, y = x.to(device), y.to(device)

            logits = net(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (step + 1) % 10 == 0:
                print('Epoch', epoch, 'step', step + 1, 'loss', '%.4f' % loss.cpu().detach().numpy(), end='')
                net.eval()
                correct = 0.
                for x, y in db_test:
                    x, y = x.to(device), y.to(device)

                    logits_q = net(x)

                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct += torch.eq(pred_q, y).sum().item()

                acc = correct / len(db_test.dataset)
                print('\tTest', step, 'acc', '%.4f' % acc)

        scheduler.step()

    path_save = '%s/model/lobs/%s_%s' % (PATH_BASE, NAME_DATA, NAME_BACKBONE)
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    path_model = '%s/net%d-ori_%s.pkl' % (path_save, idx, str(target).replace(' ', ''))
    torch.save(
        {
            'state_dict': net.state_dict(),
            'acc': acc,
            'target': target,
            'config': config
        },
        path_model
    )


def lobs(args, config, path_load, device):
    save_dict = torch.load(path_load)
    state_dict = save_dict['state_dict']
    target = save_dict['target']

    net = Learner(config, args.imgc, args.imgsz).to(device)
    net.load_state_dict(state_dict)

    # Load data
    data = DATA_SINGLE(PATH_DATA, args.imgsz, 'train', target)
    db_hessian = DataLoader(data, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)
    db = DataLoader(data, batch_size=24, shuffle=True, num_workers=8, pin_memory=True)

    data_test = DATA_SINGLE(PATH_DATA, args.imgsz, 'test', target)
    db_test = DataLoader(data_test, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    if 'masks_dict' in save_dict.keys():
        masks_dict = save_dict['masks_dict']
    else:
        masks_dict = None

    # Prune
    for step_prune, alpha in enumerate(args.cr_steps):
        # Obtain hessians
        hessians_inv = dict()
        for layer_name in net.layers_name:
            hessian_inv = generate_hessian_inv_Woodbury(net, db_hessian, layer_name, layer_name[0].upper())
            hessians_inv[layer_name] = hessian_inv

        # prune
        weights_map = net.weights_map
        state_dict = net.state_dict()
        for name in net.layers_name:
            # get name for w and b
            name_w, name_b = weights_map[name]

            # TODO: shotcut不进行剪枝
            if name.startswith('cds'):
                continue

            if name.startswith('c'):
                w = unfold(state_dict[name_w].cpu().numpy())
                kernel_shape = state_dict[name_w].shape
                b = state_dict[name_b].cpu().numpy()
                wb = np.concatenate([w, b.reshape(1, -1)], axis=0)

                if masks_dict is not None and name_w in masks_dict.keys():
                    mask_w = unfold(masks_dict[name_w])
                    mask_b = masks_dict[name_b]
                    mask_wb = np.concatenate([mask_w, mask_b.reshape(1, -1)], axis=0)
                else:
                    mask_wb = np.ones(wb.shape, dtype=np.float32)

            elif name.startswith('f'):
                w = state_dict[name_w].cpu().numpy()
                b = state_dict[name_b].cpu().numpy()
                wb = np.hstack([w, b.reshape(-1, 1)]).transpose()

                if masks_dict is not None and name_w in masks_dict.keys():
                    mask_w = masks_dict[name_w]
                    mask_b = masks_dict[name_b]
                    mask_wb = np.hstack([mask_w, mask_b.reshape(-1, 1)]).transpose()
                else:
                    mask_wb = np.ones(wb.shape, dtype=np.float32)

            # obtain arg sort
            hessian_inv = hessians_inv[name]
            l1, l2 = wb.shape
            L = np.zeros([l1 * l2])
            for row_idx in range(l1):
                for col_idx in range(l2):
                    L[row_idx * l2 + col_idx] = np.power(wb[row_idx, col_idx], 2) / (
                            hessian_inv[row_idx, row_idx] + 10e-6)
            sen_rank = np.argsort(L).tolist()

            # remove weight that already been deleted
            if masks_dict is not None and mask_wb is not None:
                for row_idx in range(l1):
                    for col_idx in range(l2):
                        if mask_wb[row_idx, col_idx] == 0:
                            sen_rank.remove(row_idx * l2 + col_idx)

            n_all = np.prod(mask_wb.shape)
            for i in range(int(alpha * n_all)):
                prune_idx = sen_rank[i]
                prune_row_idx = int(prune_idx / l2)
                prune_col_idx = prune_idx % l2
                try:
                    delta_W = -wb[prune_row_idx, prune_col_idx] / (
                            hessian_inv[prune_row_idx, prune_row_idx] + 10e-6) * hessian_inv[:, prune_row_idx]
                except Warning:
                    print('Nan found, please change another Hessian inverse calculation method')
                    break
                wb[:, prune_col_idx] += delta_W
                mask_wb[prune_row_idx, prune_col_idx] = 0

            wb = np.multiply(wb, mask_wb)
            if name.startswith('c'):
                kernel = fold_weights(wb[0:-1, :], kernel_shape)
                bias = wb[-1, :]
                mask_w = fold_weights(mask_wb[0:-1, :], kernel_shape)
                mask_b = mask_wb[-1, :]
            elif name.startswith('f'):
                kernel = wb[0:-1, :].transpose()
                bias = wb[-1, :].transpose()
                mask_w = mask_wb[0:-1, :].transpose()
                mask_b = mask_wb[-1, :].transpose()

            if masks_dict is None:
                masks_dict = dict()
                masks_dict[name_w] = mask_w
                masks_dict[name_b] = mask_b
            else:
                masks_dict[name_w] = mask_w
                masks_dict[name_b] = mask_b

            state_dict[name_w] = torch.tensor(kernel, dtype=torch.float32, requires_grad=True).to(device)
            state_dict[name_b] = torch.tensor(bias, dtype=torch.float32, requires_grad=True).to(device)

        net.load_state_dict(state_dict)

        test_masks(masks_dict, net.state_dict())

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

        cr = get_cr(net, masks_dict)
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

    # torch.manual_seed(222)
    # torch.cuda.manual_seed_all(222)
    # np.random.seed(222)

    device = torch.device('cuda')

    # 获得config
    config = get_backbone(backbone, n_way, imgsz, imgsz)

    # 获得dataset
    global DATA, DATA_SINGLE, PATH_DATA, N_CLS, PATH_BASE, NAME_DATA, NAME_BACKBONE
    DATA, DATA_SINGLE, PATH_DATA, N_CLS, PATH_BASE = get_dataset(dataset)
    NAME_DATA = dataset
    NAME_BACKBONE = backbone

    if task == 'train_net':
        train_net(args, config, device, 1)
    elif task == 'lobs':
        lobs(args, config, path_model, device)
