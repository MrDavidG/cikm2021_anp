import sys

sys.path.append('..')

from methods.anp_reptile import exp

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == '__main__':
    tasks_train_reptile = [
        {
            'n_way': 5,
            'k_spt': 1,
            'k_train': 1,
            'dataset': 'mini-imagenet',
            'backbone': 'conv32_br',
            'task': 'train_reptile',

            'log_print': True,
            'outer_lr': 1e-4
        }

    ]

    tasks_eval_reptile = [
        {
            'n_way': 5,
            'k_spt': 1,
            'k_train': 4,
            'k_qry': 1,
            'dataset': 'mini-imagenet',
            'backbone': 'conv32_br',
            'task': 'eval_reptile',

            'log_print': True,
            'path_model': '/local/home/david/Remote/VibNet_Pytorch/model/reptile_lobs/mini-imagenet_conv32_br/5-way_1-shot/net_cr-1_acc-0.3097.pkl'
        }
    ]

    tasks_anp_reptile = [

    ]

    for task in tasks_train_reptile:
        print(task)
        exp(
            n_way=task['n_way'],
            k_spt=task['k_spt'],
            k_train=task['k_train'],
            dataset=task['dataset'],
            backbone=task['backbone'],
            task=task['task'],
            path_model=task.get('path_model', None),

            k_qry=task.get('k_qry', 15),
            cr_steps=task.get('cr_steps', [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05]),
            epoch_test=task.get('epoch_test', 5),
            epoch=task.get('epoch', 100000),  # n_epoch *
            imgsz=task.get('imgsz', 84),
            imgc=task.get('imgc', 3),
            task_num=task.get('task_num', 4),
            outer_lr=task.get('outer_lr', 1e-3),
            inner_lr=task.get('inner_lr', 1e-2),
            update_step=task.get('update_step', 5),
            update_step_test=task.get('update_step_test', 5),
            log_print=task.get('log_print', True)
        )
