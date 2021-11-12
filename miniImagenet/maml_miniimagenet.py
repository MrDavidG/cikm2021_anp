import sys

sys.path.append(r"/local/home/david/Remote/VibNet_Pytorch")

from torch.utils.data import DataLoader
from dataset import mini_Imagenet
from maml import Meta

import numpy as np

import argparse
import torch
import os

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='total number of task sets', default=60000)
    argparser.add_argument('--epoch_test', type=int, help='number of test times', default=5)
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

    print(args)

    device = torch.device('cuda')

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

    maml = Meta(args, config).to(device)

    torch.save(
        {
            'state_dict': maml.net.state_dict(),
            'config': config,
            'args': args
        },
        '/local/home/david/Remote/VibNet_Pytorch/model/maml_lobs/mini-Imagenet/%d-way_%d-shot/test.pkl' % (
        args.n_way, args.k_spt)
    )

    for epoch in range(args.epoch // 10000):
        # number of meta training epochs
        mini = mini_Imagenet('/local/home/david/Remote/mini-imagenet', mode='train', n_way=args.n_way,
                             k_shot=args.k_spt,
                             k_query=args.k_qry, batchsz=10000, resize=args.imgsz)
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

        # total batchsz (default 10000) tasks, real batch size is task_num (4)
        # so there is 10000/4=2500 steps
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            accs = maml(x_spt, y_spt, x_qry, y_qry)

            # if (step + 1) % 100 == 0:
            #     print('epoch:', epoch, 'step:', step + 1, '\ttraining acc:', accs)

    torch.save(
        {
            'state_dict': maml.net.state_dict(),
            'config': config,
            'args': args
        },
        '/local/home/david/Remote/VibNet_Pytorch/model/maml_lobs/mini-Imagenet/%d-way_%d-shot/net_cr-1.pkl' % (args.n_way, args.k_spt)
    )

    for epoch in range(args.epoch_test):
        # there is batchsz(100) N-way tasks, actual batch size is 1
        # so there is 100/1=100 steps in test
        mini_test = mini_Imagenet('/local/home/david/Remote/mini-imagenet', mode='test',
                                  n_way=args.n_way,
                                  k_shot=args.k_spt,
                                  k_query=args.k_qry,
                                  batchsz=100,
                                  resize=args.imgsz)

        db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
        accs_all_test = []

        for x_spt, y_spt, x_qry, y_qry in db_test:
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(
                device), x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

            accs, losses = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
            accs_all_test.append(accs)

        # [b(100/1=100), update_step+1]
        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
        print('Test acc:', accs)
        print('Test losses:', ['{:.4f}'.format(_) for _ in losses])
