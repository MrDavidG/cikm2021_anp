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


class Meta(Module):
    """
    Meta Learner
    """

    def __init__(self, args, config, learner=None):
        """

        :param args:
        :param config:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.meta_decay = args.meta_decay
        self.meta_decay_step_size = args.meta_decay_step_size
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

        self.meta_optim = Adam(self.net.parameters(), None, lr=self.meta_lr)
        # TODO:
        if self.meta_decay < 1:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.meta_optim, self.meta_decay_step_size,
                                                             self.meta_decay)

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

    def forward(self, x_spt, y_spt, x_qry, y_qry, masks_dict, log=False):
        """

        :param x_spt: [task_num, setsz, c, h, w]
        :param y_spt: [task_num, setsz]
        :param x_qry: [task_num, querysz, c, h, w]
        :param y_qry: [b, querysz]
        :return:
        """

        # 这里重新创建，因为masks_dict是变化的
        self.meta_optim.set_masks(masks_dict)

        task_num, setsz, c, h, w = x_spt.size()
        querysz = x_qry.size(1)

        corrects = [0 for _ in range(self.update_step + 1)]
        grads = [0 for _ in self.net.parameters()]
        # loss_sum = 0

        for i in range(task_num):
            # run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters(), retain_graph=SECOND, create_graph=SECOND)
            grad = mask_grad(grad, masks_dict)

            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # 如果不输出accs的话这里是不需要的，不涉及梯度更新
            if log:
                # this is the loss and accuracy before first update
                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)

                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()
                    corrects[0] = corrects[0] + correct

                # the loss and accuracy after the first update
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)

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

                # the final loss on the qry data, used to update theta
                if k == self.update_step - 1 or log:
                    logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                    if k == self.update_step - 1:
                        loss_q = F.cross_entropy(logits_q, y_qry[i])
                        # loss_sum += loss_q
                    if log:
                        with torch.no_grad():
                            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                            correct = torch.eq(pred_q, y_qry[i]).sum().item()
                            corrects[k + 1] = corrects[k + 1] + correct

            # 在这里就计算梯度，释放计算图才能保证继续计算，要的就是loss_q，目标是得到梯度，但是不更新，还需要是adam的梯度
            # self.meta_optim.zero_grad()
            # loss_q.backward()
            # grads = [p.grad.data.detach() + grad for p, grad in zip(self.net.parameters(), grads)]
            # self.meta_optim.zero_grad()

            # 这里是求出对当前task的所有update后的二阶梯度
            grads_task = torch.autograd.grad(loss_q, self.net.parameters())
            grads = [grad_p.detach() + grad for grad_p, grad in zip(grads_task, grads)]
            self.meta_optim.zero_grad()

        # average grad from all tasks
        # grads = [grad / task_num for grad in grads]
        # self.meta_optim.step_with_grads(grads)

        self.meta_optim.zero_grad()
        for p, g in zip(self.net.parameters(), grads):
            p.grad = g / float(task_num)
            # p.grad.data.clamp_(-10, 10)
        self.meta_optim.step()

        if self.meta_decay < 1:
            self.scheduler.step()

        # type 2
        # loss = loss_sum / task_num
        # self.meta_optim.zero_grad()
        # loss.backward()
        # self.meta_optim.step()

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

        # TODO: 为了调参
        lr = self.update_lr

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
        fast_weights = list(map(lambda p: p[1] - lr * p[0], zip(grad, net.parameters())))

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

            fast_weights = list(map(lambda p: p[1] - lr * p[0], zip(grad, fast_weights)))

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


def maml_lobs(args, config, path_load=None, device=None):
    if path_load is not None:
        print('Load net from', path_load)
        save_dict = torch.load(path_load)
        if 'accs' in save_dict.keys():
            print('Accs', save_dict['accs'])
        if 'losses' in save_dict.keys():
            print('Losses', save_dict['losses'])

        state_dict = save_dict['state_dict']
        learner = Learner(config, args.imgc, args.imgsz)
        learner.load_state_dict(state_dict)
        if 'masks_dict' in save_dict.keys():
            masks_dict = save_dict['masks_dict']
        else:
            print('Masks_dict is initialized as None')
            masks_dict = None

        test_masks(masks_dict, state_dict)
    else:
        learner = None
        masks_dict = None

    maml = Meta(args, config, learner).to(device)
    for idx_prune, cr_step in enumerate(args.cr_steps):
        # calculate hessians
        hessians_inv = dict()
        for layer_name in maml.net.layers_name:
            # kernel size
            if layer_name.startswith('cds'):
                layer_kernel = 1
            else:
                layer_kernel = 3

            if True:
                # fsl setting
                data = DATA(PATH_DATA, mode='train', n_way=args.n_way, k_shot=args.k_spt,
                            k_query=args.k_qry, batchsz=10000, resize=args.imgsz)
                db_hessian = DataLoader(data, 20, shuffle=True, num_workers=8, pin_memory=True)
                hessian_inv = generate_hessian_inv_Woodbury_maml_fsl(maml.net, db_hessian, layer_name,
                                                                     layer_name[0].upper(), device, mask_grad,
                                                                     masks_dict, args.update_step, args.update_lr,
                                                                     SECOND, stride_factor=5, layer_kernel=layer_kernel)
            else:
                pass
                # all instances
                # data = mini_Imagenet_hessian(PATH_DATA, mode='train', n_way=args.n_way, k_shot=40, k_query=0,
                #                              batchsz=50, resize=args.imgsz)
                # # 必须是False
                # db_hessian = DataLoader(data, 20, shuffle=False, num_workers=8, pin_memory=True)
                # hessian_inv = generate_hessian_inv_Woodbury_maml_all(maml.net, db_hessian, layer_name,
                #                                                      layer_name[0].upper(),
                #                                                      device, mask_grad, masks_dict, args.update_ste,
                #                                                      args.update_lr, SECOND, stride_factor=5)
            hessians_inv[layer_name] = hessian_inv

        # LOBS prune
        weights_map = maml.net.weights_map

        state_dict = maml.net.state_dict()
        for name in maml.net.layers_name:
            name_w, name_b = weights_map[name]
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
            for i in range(int(cr_step * n_all)):
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

        maml.net.load_state_dict(state_dict)
        cr = get_cr(maml.net, masks_dict)
        print('N_prune', idx_prune, 'CR', cr)

        # mask_dict和weight是否有效
        test_masks(masks_dict, state_dict)

        # Test after prune
        meta_test(maml, args, masks_dict, device)
        print('Start to Retrain ...')

        path_save = '%s/model/maml_lobs/%s_%s/%d-way_%d-shot' % (
            PATH_BASE, NAME_DATA, NAME_BACKBONE, args.n_way, args.k_spt)
        if not os.path.exists(path_save):
            os.makedirs(path_save)

        # retrain
        meta_train(maml, args, masks_dict, device,
                   output=args.log_print,
                   save=args.save_each_epoch,
                   save_threshold=args.save_threshold,
                   test_each_epoch=args.test_each_epoch,
                   path_save=path_save,
                   cr=cr,
                   break_threshold=args.break_threshold)

        # Test after retrain
        acc_avg, accs = meta_test(maml, args, masks_dict, device)

        test_masks(masks_dict, state_dict)

        path_model = '%s/net_cr-%.2f_acc-%.4f.pkl' % (path_save, cr, acc_avg)
        print('Save as %s' % path_model)
        torch.save(
            {
                'state_dict': maml.net.state_dict(),
                'masks_dict': masks_dict,
                'accs': accs,
                'config': config
            },
            path_model
        )


def meta_test(maml, args, masks_dict, device, once=False):
    global DATA, PATH_DATA

    acc_max_list = list()
    if once:
        n_epoch = 1
    else:
        n_epoch = args.epoch_test
    for epoch in range(n_epoch):
        data_test = DATA(PATH_DATA, mode='test', n_way=args.n_way, k_shot=args.k_spt,
                         k_query=args.k_qry, batchsz=100, resize=args.imgsz)
        db_test = DataLoader(data_test, 1, shuffle=True, num_workers=1, pin_memory=True)

        accs_all_test = []
        for x_spt, y_spt, x_qry, y_qry in db_test:
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(
                device), x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

            accs, losses = maml.finetunning(x_spt, y_spt, x_qry, y_qry, masks_dict)
            accs_all_test.append(accs)

        # [b(100/1=100), update_step+1]
        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16).tolist()
        print('Test acc:', '[' + ', '.join(['{:.4f}'.format(_) for _ in accs]) + ']')
        print('Test losses:', '[' + ', '.join(['{:.4f}'.format(_) for _ in losses]) + ']')
        acc_max_list.append(np.max(accs))
    return np.mean(acc_max_list), accs


def meta_train(maml, args, masks_dict, device, output=False, save=True, save_threshold=0., path_save=None,
               test_each_epoch=False, cr=1., break_threshold=1.1):
    global DATA, PATH_DATA

    for epoch in range(args.epoch // 10000):
        # number of meta training epochs
        data = DATA(PATH_DATA, mode='train', n_way=args.n_way, k_shot=args.k_train, k_query=args.k_qry, batchsz=10000,
                    resize=args.imgsz)
        db = DataLoader(data, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

        # total batchsz (default 10000) tasks, real batch size is task_num (4)
        # so there is 10000/4=2500 steps
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            # 这里是进行一次以set为单位的adaptation
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            accs = maml(x_spt, y_spt, x_qry, y_qry, masks_dict, (step + 1) % 500 == 0)

            if output and (step + 1) % 500 == 0:
                print('epoch:', epoch, 'step:', step + 1, '\ttraining acc:', accs)

        acc_avg, accs = meta_test(maml, args, masks_dict, device, once=not test_each_epoch or not save)

        if save and acc_avg > save_threshold:
            path_model = '%s/net_cr-%.2f_acc-%.4f.pkl' % (path_save, cr, acc_avg)
            print('Save as %s' % path_model)
            torch.save(
                {
                    'state_dict': maml.net.state_dict(),
                    'masks_dict': masks_dict,
                    'accs': accs,
                    'args': args
                },
                path_model
            )

        if acc_avg >= break_threshold:
            print('Train accuracy', acc_avg, 'is larger than', break_threshold, '. Training is break!')
            break


def eval_maml(args, config, path_load, device):
    print('Load net from', path_load)
    save_dict = torch.load(path_load)

    if 'accs' in save_dict.keys():
        print('Accs', save_dict['accs'])
    if 'losses' in save_dict.keys():
        print('Losses', save_dict['losses'])

    state_dict = save_dict['state_dict']
    learner = Learner(config, args.imgc, args.imgsz)
    learner.load_state_dict(state_dict)
    if 'masks_dict' in save_dict.keys():
        masks_dict = save_dict['masks_dict']
    else:
        masks_dict = None

    test_masks(masks_dict, state_dict)
    maml = Meta(args, config, learner).to(device)

    meta_test(maml, args, masks_dict, device)


def train_maml(args, config, path_model, device):
    """
    used to train a maml net
    :param args:
    :param config:
    :return:
    """
    global DATA, PATH_BASE, PATH_DATA, N_CLS, NAME_DATA, NAME_BACKBONE

    if path_model is None:
        masks_dict = None
        learner = None
    else:
        print('Load net from', path_model)
        save_dict = torch.load(path_model)
        state_dict = save_dict['state_dict']
        masks_dict = save_dict.get('masks_dict', None)

        learner = Learner(config, args.imgc, args.imgsz)
        learner.load_state_dict(state_dict)

    if masks_dict is None:
        print('masks_dict is None!')
        cr = 1.
    else:
        cr = get_cr(learner, masks_dict)

    maml = Meta(args, config, learner).to(device)

    # save path
    if path_model is None:
        path_save = '%s/model/maml_lobs/%s_%s/%d-way_%d-shot' % (
            PATH_BASE, NAME_DATA, NAME_BACKBONE, args.n_way, args.k_spt)
    else:
        path_save = '/'.join(path_model.split('/')[:-1])
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    meta_test(maml, args, masks_dict, device)
    # meta train
    meta_train(maml, args, masks_dict, device,
               output=args.log_print,
               save=args.save_each_epoch,
               save_threshold=args.save_threshold,
               test_each_epoch=args.test_each_epoch,
               path_save=path_save,
               cr=cr,
               break_threshold=args.break_threshold)

    if not args.save_each_epoch:
        # meta test
        acc_avg, accs = meta_test(maml, args, masks_dict, device)

        path_model = '%s/net_cr-%.2f_acc-%.4f.pkl' % (path_save, cr, acc_avg)
        print('Save as %s' % path_model)
        torch.save(
            {
                'state_dict': maml.net.state_dict(),
                'accs': accs,
                'config': config,
                'args': args
            },
            path_model
        )


def exp(n_way,
        k_train,
        k_spt,
        dataset,
        backbone,
        task,
        path_model=None,
        k_qry=15,
        cr_steps=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05],
        epoch_test=5,
        epoch=60000,
        imgsz=84,
        imgc=3,
        task_num=4,
        meta_decay=1,
        meta_decay_step_size=2500,
        meta_lr=1e-3,
        update_lr=0.01,
        update_step=5,
        update_step_test=10,
        save_each_epoch=False,
        save_threshold=0.,
        break_threshold=1.2,
        test_each_epoch=False,
        log_print=False):
    argparser = argparse.ArgumentParser()
    # prune
    argparser.add_argument('--cr_steps', type=list, help='cr at each iteration', default=cr_steps)
    argparser.add_argument('--epoch_test', type=int, help='number of test times', default=epoch_test)
    # maml
    argparser.add_argument('--log_print', type=bool, help='print logs for meta training', default=log_print)
    argparser.add_argument('--epoch', type=int, help='total number of task sets', default=epoch)
    argparser.add_argument('--n_way', type=int, help='n way', default=n_way)
    argparser.add_argument('--k_train', type=int, help='n train', default=k_train)
    argparser.add_argument('--save_each_epoch', type=bool, help='save in each epoch', default=save_each_epoch)
    argparser.add_argument('--save_threshold', type=float, help='save threshold in maml', default=save_threshold)
    argparser.add_argument('--break_threshold', type=float, help='break threshold in cavia', default=break_threshold)
    argparser.add_argument('--test_each_epoch', type=float, help='test each epoch', default=test_each_epoch)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=k_spt)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=k_qry)
    argparser.add_argument('--imgsz', type=int, help='image size', default=imgsz)
    argparser.add_argument('--imgc', type=int, help='number of channels for images', default=imgc)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=task_num)
    argparser.add_argument('--meta_decay', type=float, help='decay of outer learning rate', default=meta_decay)
    argparser.add_argument('--meta_decay_step_size', type=int, help='step size for meta decay',
                           default=meta_decay_step_size)
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

    # 运行相应的task
    if task == 'train_maml':
        train_maml(args, config, path_model, device)
    elif task == 'maml_lobs':
        maml_lobs(args, config, path_model, device)
    elif task == 'eval_maml':
        eval_maml(args, config, path_model, device)
    else:
        print('Cannot find task!')
        raise RuntimeError
