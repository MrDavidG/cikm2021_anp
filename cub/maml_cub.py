import sys

sys.path.append(r"/local/home/david/Remote/VibNet_Pytorch")

from torch.utils.data import DataLoader
from dataset import CUB
from maml import Meta

import numpy as np

import argparse
import torch
import os


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='total number of task sets', default=60000)
    argparser.add_argument('--epoch_test', type=int, help='number of test times', default=10)
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
    # br
    config = [
        ('conv2d', [32, 3, 3, 3, 1, 1]),
        ('bn', [32]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('bn', [32]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('bn', [32]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('bn', [32]),
        ('relu', [True]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    maml = Meta(args, config).to(device)

    for epoch in range(args.epoch // 10000):
        # number of meta training epochs
        data = CUB('/local/home/david/Downloads/cub/CUB_200_2011', mode='train', n_way=args.n_way,
                   k_shot=args.k_spt,
                   k_query=args.k_qry, batchsz=10000, resize=args.imgsz)
        db = DataLoader(data, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

        # total batchsz (default 10000) tasks, real batch size is task_num (4)
        # so there is 10000/4=2500 steps
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            accs = maml(x_spt, y_spt, x_qry, y_qry)

    acc_max_list = list()
    for epoch in range(args.epoch_test):
        # there is batchsz(100) N-way tasks, actual batch size is 1
        # so there is 100/1=100 steps in test
        data_test = CUB('/local/home/david/Downloads/cub/CUB_200_2011', mode='test',
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

            accs, losses = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
            accs_all_test.append(accs)

        # [b(100/1=100), update_step+1]
        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
        print('Test acc:', '[' + ', '.join(['{:.4f}'.format(_) for _ in accs]) + ']')
        print('Test losses:', '[' + ', '.join(['{:.4f}'.format(_) for _ in losses]) + ']')

        acc_max_list.append(np.max(accs))

    path_save = '/local/home/david/Remote/VibNet_Pytorch/model/maml_lobs/cub_conv32_br/5-way_5-shot'
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    path_model = '%s/net_cr-1_acc-%.4f.pkl' % (path_save, np.mean(acc_max_list))
    print('Save as %s' % path_model)
    torch.save(
        {
            'state_dict': maml.net.state_dict(),
            'config': config,
            'args': args
        },
        path_model
    )
