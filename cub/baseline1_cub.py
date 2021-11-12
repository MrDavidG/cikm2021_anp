import sys

sys.path.append(r"/local/home/david/Remote/VibNet_Pytorch")

# from utils_hessian import generate_conv_hessian, generate_fc_hessian, extract_image_patches
from components.utils_hessian_tf import generate_hessian, generate_hessian_inv_Woodbury
from torch.nn import Module
from components.optimizer import Adam
from learner import Learner
from copy import deepcopy
from dataset import CUB, CUB_one_task
from torch.utils.data import DataLoader
from numpy.linalg import inv, pinv, LinAlgError

import torch.nn.functional as F
import numpy as np

import argparse
import torch
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


PATH_CUB = '/local/home/david/Downloads/cub/CUB_200_2011'
N_CLS_CUB = 100
SECOND = True


def test_masks(masks_dict, state_dict):
    masks = trans_masks(masks_dict)
    if masks is not None:
        if len(masks) == 0:
            print('length of masks_dict is 0!')
            raise RuntimeError
        for k in masks.keys():
            v = torch.sum(
                (torch.tensor(masks[k] == 0).float().cuda() + (state_dict[k] != 0).float()) == 2).cpu().numpy()
            # print(k, v)
            if v != 0:
                print(v)
                raise RuntimeError
    else:
        print('masks_dict is None!')
    print('masks_dict and weight is OK!')


def trans_masks(masks):
    if masks is None:
        return None
    else:
        trans = [('vars.0', 'c1.weight'),
                 ('vars.1', 'c1.bias'),
                 ('vars.2', 'b1.weight'),
                 ('vars.3', 'b1.bias'),
                 ('vars.4', 'c2.weight'),
                 ('vars.5', 'c2.bias'),
                 ('vars.6', 'b2.weight'),
                 ('vars.7', 'b2.bias'),
                 ('vars.8', 'c3.weight'),
                 ('vars.9', 'c3.bias'),
                 ('vars.10', 'b3.weight'),
                 ('vars.11', 'b3.bias'),
                 ('vars.12', 'c4.weight'),
                 ('vars.13', 'c4.bias'),
                 ('vars.14', 'b4.weight'),
                 ('vars.15', 'b4.bias'),
                 ('vars.16', 'f5.weight'),
                 ('vars.17', 'f5.bias')]
        masks_ = dict()
        for (n, m) in trans:
            if m in masks.keys():
                masks_[n] = masks[m]
        return masks_


def mask_grad(grad, masks):
    if masks is None:
        return grad

    if 'c1.weight' in masks.keys():
        # print('trans masks into vars.0')
        masks_dict = trans_masks(masks)
    elif 'vars.0' in masks.keys():
        masks_dict = masks
    else:
        print('Keys in grad masks is wrong!!!')
        raise RuntimeError

    if type(grad) is tuple:
        grad = list(grad)

    for idx, v in enumerate(grad):
        k = 'vars.%d' % idx
        if k in masks_dict.keys():
            grad[idx] = v * torch.tensor(masks_dict[k], dtype=torch.float32).cuda()
    return grad


class Meta(Module):
    """
    Meta Learner
    """

    def __init__(self, args, config, learner=None, masks_dict=None):
        """

        :param args:
        :param config:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        if learner is None:
            self.net = Learner(config, args.imgc, args.imgsz)
        else:
            self.net = learner
        self.meta_optim = Adam(self.net.parameters(), trans_masks(masks_dict), lr=self.meta_lr)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter

    def forward(self, x_spt, y_spt, x_qry, y_qry, masks_dict):
        """

        :param x_spt: [task_num, setsz, c, h, w]
        :param y_spt: [task_num, setsz]
        :param x_qry: [task_num, querysz, c, h, w]
        :param y_qry: [b, querysz]
        :return:
        """
        task_num, setsz, c, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            # run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters(), retain_graph=SECOND, create_graph=SECOND)
            grad = mask_grad(grad, masks_dict)
            # TODO
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # the loss and accuracy after the first update
            # [setsz, nway]
            logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry[i])
            losses_q[1] += loss_q

            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry[i]).sum().item()
            corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                #  2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights, retain_graph=SECOND, create_graph=SECOND)
                # 3. theta_pi = theta_pi - train_lr * grad
                # TODO
                grad = mask_grad(grad, masks_dict)

                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the losss_q on last update step
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * task_num)
        # [self.update_step + 1]
        return accs

    def finetunning(self, x_spt, y_spt, x_qry, y_qry, masks_dict):
        """

        :param x_spt: [setsz, c, h, w]
        :param y_spt: [setsz]
        :param x_qry: [querysz, c, h, w]
        :param y_qry: [querysz]
        :return: [update_step_test]
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]
        losses = [0 for _ in range(self.update_step_test + 1)]

        # in order to net ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        self.net.layers_input = None
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters(), retain_graph=SECOND, create_graph=SECOND)
        # TODO: mask
        grad = mask_grad(grad, masks_dict)
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, n_way]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct
            losses[0] = losses[0] + loss_q.cpu().detach().numpy()

        # this is the loss and accuracy after the first update
        # [setsz, n_way]
        logits_q = net(x_qry, fast_weights, bn_training=True)
        loss_q = F.cross_entropy(logits_q, y_qry)
        # [setsz]
        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        # scalar
        correct = torch.eq(pred_q, y_qry).sum().item()
        corrects[1] = corrects[1] + correct
        losses[1] = losses[1] + loss_q.cpu().detach().numpy()

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights, retain_graph=SECOND, create_graph=SECOND)
            # 3. theta_pi = theta_pi - train_lr * grad
            # TODO: mask
            grad = mask_grad(grad, masks_dict)

            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[k + 1] = corrects[k + 1] + correct
                losses[k + 1] = losses[k + 1] + loss_q.cpu().detach().numpy()

        del net

        accs = np.array(corrects) / querysz
        return accs, losses


def unfold(kernel):
    k_shape = kernel.shape
    weight = np.zeros([k_shape[1] * k_shape[2] * k_shape[3], k_shape[0]])
    for i in range(k_shape[0]):
        weight[:, i] = np.reshape(kernel[i, :, :, :], [-1])

    return weight


def get_inv(mat):
    try:
        mat_inv = inv(mat)
    except LinAlgError:
        print(LinAlgError)
        mat_inv = pinv(mat)
    return mat_inv


def fold_weights(weights, kernel_shape):
    """
    In pytorch format, kernel is stored as [out_channel, in_channel, width, height]
    Fold weights into a 4-dimensional tensor as [out_channel, in_channel, width, height]
    :param weights:
    :param kernel_shape:
    :return:
    """
    kernel = np.zeros(shape=kernel_shape)
    for i in range(kernel_shape[0]):
        kernel[i, :, :, :] = weights[:, i].reshape([kernel_shape[1], kernel_shape[2], kernel_shape[3]])

    return kernel


def get_cr(net, masks):
    sum_all = 0
    sum_zero = 0
    for p in net.parameters():
        sum_all += np.prod(p.shape)
    for _, v in masks.items():
        sum_zero += np.sum(v == 0)
    cr = (sum_all - sum_zero) / sum_all
    print('Model Summary: ')
    print('Reserve params', sum_all - sum_zero, 'All params', sum_all, 'CR', '%.4f' % cr)
    return cr


def trans_key(key):
    trans_dict = {'c1.weight': 'vars.0',
                  'c1.bias': 'vars.1',
                  'b1.weight': 'vars.2',
                  'b1.bias': 'vars.3',
                  'c2.weight': 'vars.4',
                  'c2.bias': 'vars.5',
                  'b2.weight': 'vars.6',
                  'b2.bias': 'vars.7',
                  'c3.weight': 'vars.8',
                  'c3.bias': 'vars.9',
                  'b3.weight': 'vars.10',
                  'b3.bias': 'vars.11',
                  'c4.weight': 'vars.12',
                  'c4.bias': 'vars.13',
                  'b4.weight': 'vars.14',
                  'b4.bias': 'vars.15',
                  'f5.weight': 'vars.16',
                  'f5.bias': 'vars.17'}
    return trans_dict[key]


def train_net(args, config, idx):
    device = torch.device('cuda')
    target = np.sort(np.random.choice(np.arange(N_CLS_CUB), args.n_way, replace=False)).tolist()
    print('Train net for target', target)

    # Load data
    data = CUB_one_task(PATH_CUB, args.imgsz, 'train', target)
    db = DataLoader(data, batch_size=24, shuffle=True, num_workers=8, pin_memory=True)
    data_test = CUB_one_task(PATH_CUB, args.imgsz, 'test', target)
    db_test = DataLoader(data_test, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    # Load net
    net = Learner(config, args.imgc, args.imgsz).to(device)

    for epoch in range(8):
        opt = Adam(net.parameters(), None, lr=0.01 * 0.99 ** epoch)
        for step, (x, y) in enumerate(db):
            net.train()
            x, y = x.to(device), y.to(device)

            logits = net(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (step + 1) % 10 == 0:
                print('Epoch', epoch, 'step', step + 1, 'loss', '%.4f' % loss.cpu().detach().numpy())
                net.eval()
                correct = 0.
                for x, y in db_test:
                    x, y = x.to(device), y.to(device)

                    logits_q = net(x)

                    with torch.no_grad():
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                        correct += torch.eq(pred_q, y).sum().item()

                acc = correct / len(db_test.dataset)
                print('Test', step, 'acc', '%.4f' % acc)

    torch.save({'state_dict': net.state_dict(), 'acc': acc, 'target': target, 'config': config},
               '../model/lobs/cub/net%d-ori_%s.pkl' % (idx, str(target).replace(' ', '')))


def lobs(config, path_net, args, n_prune, alpha=0.1):
    hessian_batch_size = 2

    device = torch.device('cuda')

    save_dict = torch.load(path_net)
    # config = save_dict['config']
    state_dict = save_dict['state_dict']
    target = save_dict['target']

    net = Learner(config, args.imgc, args.imgsz).to(device)
    net.load_state_dict(state_dict)

    # Load data
    data_train = CUB_one_task(PATH_CUB, args.imgsz, 'train', target)
    db_hessian = DataLoader(data_train, batch_size=hessian_batch_size, shuffle=True, num_workers=8, pin_memory=True)
    db = DataLoader(data_train, batch_size=24, shuffle=True, num_workers=8, pin_memory=True)

    data_test = CUB_one_task(PATH_CUB, args.imgsz, 'test', target)
    db_test = DataLoader(data_test, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    if 'masks_dict' in save_dict.keys():
        masks_dict = save_dict['masks_dict']
    else:
        masks_dict = None

    # Prune
    woodbury_list = ['c1', 'c2', 'c3', 'c4', 'f5']

    for step_prune in range(n_prune):
        # Obtain hessians
        hessians_inv = dict()

        for layer_name in net.layers_name:
            if layer_name.startswith('c'):
                layer_type = 'C'
            elif layer_name.startswith('f'):
                layer_type = 'F'

            if layer_name in woodbury_list:
                hessian_inv = generate_hessian_inv_Woodbury(net, db_hessian, layer_name, layer_type,
                                                            n_batch_used=100,
                                                            batch_size=hessian_batch_size,
                                                            stride_factor=5)
            else:
                hessian = generate_hessian(net, db_hessian, layer_name, layer_type,
                                           n_batch_used=100,
                                           batch_size=hessian_batch_size,
                                           stride_factor=5)
                hessian_inv = get_inv(hessian)

            hessians_inv[layer_name] = hessian_inv

        # prune
        state_dict = net.state_dict()
        for name in net.layers_name:
            if name.startswith('c'):
                w = unfold(state_dict[trans_key(name + '.weight')].cpu().numpy())
                kernel_shape = state_dict[trans_key(name + '.weight')].shape
                b = state_dict[trans_key(name + '.bias')].cpu().numpy()
                wb = np.concatenate([w, b.reshape(1, -1)], axis=0)

                if masks_dict is not None and name + '.weight' in masks_dict.keys():
                    mask_w = unfold(masks_dict[name + '.weight'])
                    mask_b = masks_dict[name + '.bias']
                    mask_wb = np.concatenate([mask_w, mask_b.reshape(1, -1)], axis=0)
                else:
                    mask_wb = np.ones(wb.shape, dtype=np.float32)

            elif name.startswith('f'):
                w = state_dict[trans_key(name + '.weight')].cpu().numpy()
                b = state_dict[trans_key(name + '.bias')].cpu().numpy()
                wb = np.hstack([w, b.reshape(-1, 1)]).transpose()

                if masks_dict is not None and name + '.weight' in masks_dict.keys():
                    mask_w = masks_dict[name + '.weight']
                    mask_b = masks_dict[name + '.bias']
                    mask_wb = np.hstack([mask_w, mask_b.reshape(-1, 1)]).transpose()
                else:
                    mask_wb = np.ones(wb.shape, dtype=np.float32)

            # obtain arg sort
            # obtain hessian_inv
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
                    print('Nan found, please change another Hessian inverse calculation method', name)
                    break
                wb[:, prune_col_idx] += delta_W
                mask_wb[prune_row_idx, prune_col_idx] = 0

            wb = np.multiply(wb, mask_wb)

            if name.startswith('c'):
                kernel = fold_weights(wb[0:-1, :], kernel_shape)
                bias = wb[-1, :]
                mask_w = fold_weights(mask_wb[0:-1, :], kernel_shape)
                mask_b = mask_wb[-1, :]
            else:
                kernel = wb[0:-1, :].transpose()
                bias = wb[-1, :].transpose()
                mask_w = mask_wb[0:-1, :].transpose()
                mask_b = mask_wb[-1, :].transpose()

            if masks_dict is None:
                masks_dict = dict()
                masks_dict[name + '.weight'] = mask_w
                masks_dict[name + '.bias'] = mask_b
            else:
                masks_dict[name + '.weight'] = mask_w
                masks_dict[name + '.bias'] = mask_b

            state_dict[trans_key(name + '.weight')] = torch.tensor(kernel, dtype=torch.float32,
                                                                   requires_grad=True).to(device)
            state_dict[trans_key(name + '.bias')] = torch.tensor(bias, dtype=torch.float32,
                                                                 requires_grad=True).to(device)
        net.load_state_dict(state_dict)

        # Retrain
        for epoch in range(20):
            opt = Adam(net.parameters(), trans_masks(masks_dict), lr=0.01 * 0.99 ** epoch)
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
                    print('Epoch', epoch, 'step', step + 1, 'loss', '%.4f' % loss.cpu().detach().numpy(), 'Test', step,
                          'acc', '%.4f' % acc)

        # Save
        test_masks(masks_dict, net.state_dict())
        if not os.path.exists(path_net.split('-')[0]):
            os.mkdir(path_net.split('-')[0])
        cr = get_cr(net, masks_dict)
        torch.save({'state_dict': net.state_dict(), 'masks_dict': masks_dict, 'target': target, 'cr': cr},
                   path_net.split('-')[0] + '/net-cr_%.2f.pkl' % cr)


def maml(args, config, path_load=None, reinit=False):
    device = torch.device('cuda')

    learner = Learner(config, args.imgc, args.imgsz)

    if path_load is not None:
        print('Load net from', path_load)
        save_dict = torch.load(path_load)
        masks_dict = save_dict['masks_dict']
        state_dict = save_dict['state_dict']

        if reinit:
            learn_init = Learner(config, args.imgc, args.imgsz).to(device)
            state_dict_init = learn_init.state_dict()
            del learn_init

            for k, mask in masks_dict.items():
                state_dict_init[trans_key(k)] = state_dict_init[trans_key(k)] * torch.tensor(mask,
                                                                                             dtype=torch.float32).cuda()
            state_dict = state_dict_init
            print('Reinit weights！！！')

        test_masks(masks_dict, state_dict)

        learner.load_state_dict(state_dict)
    else:
        masks_dict = None

    maml = Meta(args, config, learner, masks_dict).to(device)

    for epoch in range(args.epoch // 10000):
        # number of meta training epochs
        data = CUB(PATH_CUB, mode='train', n_way=args.n_way,
                   k_shot=args.k_spt,
                   k_query=args.k_qry, batchsz=10000, resize=args.imgsz)
        db = DataLoader(data, args.task_num, shuffle=True, num_workers=8, pin_memory=True)

        # total batchsz (default 10000) tasks, real batch size is task_num (4)
        # so there is 10000/4=2500 steps
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            maml(x_spt, y_spt, x_qry, y_qry, masks_dict)

            # if (step + 1) % 1000 == 0:
            #     test_masks(masks_dict, maml.net.state_dict())

    for epoch in range(args.epoch_test):
        # there is batchsz(100) N-way tasks, actual batch size is 1
        # so there is 100/1=100 steps in test
        data_test = CUB(PATH_CUB, mode='test',
                        n_way=args.n_way,
                        k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=100,
                        resize=args.imgsz)

        db_test = DataLoader(data_test, 1, shuffle=True, num_workers=1, pin_memory=True)
        accs_all_test = []

        for x_spt, y_spt, x_qry, y_qry in db_test:
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(
                device), x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

            accs, losses = maml.finetunning(x_spt, y_spt, x_qry, y_qry, masks_dict)
            accs_all_test.append(accs)

        # [b(100/1=100), update_step+1]
        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
        print('Test acc:', '[' + ', '.join(['{:.4f}'.format(_) for _ in accs]) + ']')
        print('Test losses:', '[' + ', '.join(['{:.4f}'.format(_) for _ in losses]) + ']')

    test_masks(masks_dict, maml.net.state_dict())


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    # prune
    argparser.add_argument('--cr_step', type=bool, help='cr at each iteration', default=0.1)
    # model
    argparser.add_argument('--n_hidden', type=int, help='dimension of the network', default=32)
    # maml
    argparser.add_argument('--epoch', type=int, help='total number of task sets', default=60000)
    argparser.add_argument('--epoch_test', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='image size', default=84)
    argparser.add_argument('--imgc', type=int, help='number of channels for images', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for fine tunning', default=10)

    args = argparser.parse_args()

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    a = [32, 32, 32, 32]
    config = [
        ('conv2d', [a[0], 3, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [a[0]]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [a[1], a[0], 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [a[1]]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [a[2], a[1], 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [a[2]]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [a[3], a[2], 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [a[3]]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [args.n_way, a[3] * 5 * 5])
    ]

    # maml
    for path in [
        ''
    ]:
        maml(args, config, path, reinit=False)
