import sys

sys.path.append('..')

from methods.anp_cavia import exp

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == '__main__':

    tasks_train_cavia = [
        # mini-Imagenet conv32_br 5-way 1-shot
        {
            'n_way': 5,
            'k_spt': 1,
            'k_train': 1,
            'dataset': 'mini-imagenet',
            'backbone': 'conv32_br',
            'task': 'train_cavia',
            'epoch': 400000,

            'path_model': '/local/home/david/Remote/VibNet_Pytorch/model/cavia_lobs/mini-imagenet_conv32_br/5-way_1-shot/net_cr-1.00_acc-0.4556.pkl',
            'inner_lr': 0.12,
            'test_each_epoch': True,
            'save_each_epoch': True,
            'log_print': True
        }
    ]


    for task in tasks_train_cavia:
        print(task)
        exp(
            n_way=task['n_way'],
            k_spt=task['k_spt'],
            k_train=task['k_train'],
            dataset=task['dataset'],
            backbone=task['backbone'],
            task=task['task'],
            path_model=task.get('path_model', None),
            num_context_params=task.get('num_context_params', 100),
            context_in=task.get('context_in', [False, False, True, False, False]),
            k_qry=task.get('k_qry', 15),
            cr_steps=task.get('cr_steps', [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05]),
            epoch_test=task.get('epoch_test', 5),
            epoch=task.get('epoch', 200000),
            imgsz=task.get('imgsz', 84),
            imgc=task.get('imgc', 3),
            task_num=task.get('task_num', 16),
            outer_lr=task.get('outer_lr', 1e-3),
            inner_lr=task.get('inner_lr', 1.),
            update_step=task.get('update_step', 2),
            update_step_test=task.get('update_step_test', 4),
            save_each_epoch=task.get('save_each_epoch', False),
            save_threshold=task.get('save_threshold', 0),
            break_threshold=task.get('break_threshold', 1.2),
            test_each_epoch=task.get('test_each_epoch', False),
            log_print=task.get('log_print', False)
        )
