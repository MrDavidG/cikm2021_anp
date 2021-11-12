import sys

sys.path.append('..')

from methods.anp_maml import exp

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == '__main__':
    tasks_train_maml = [
        # mini-Imagenet resnet12 5-way 1-shot
        {
            'n_way': 5,
            'k_spt': 1,
            'k_train': 1,
            'dataset': 'mini-imagenet',
            'backbone': 'resnet12',
            'task': 'train_maml',
            'epoch': 100000,

            'test_each_epoch': True,
            'save_each_epoch': True,
            'save_threshold': 0.5,

            'path_model': '/home/david/Remote/VibNet_Pytorch/model/maml_lobs/mini-imagenet_resnet12/5-way_1-shot/net_cr-1_acc-0.4898.pkl',

            'log_print': True
        }
    ]

    for task in tasks_train_maml:
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
            epoch=task.get('epoch', 60000),
            imgsz=task.get('imgsz', 84),
            imgc=task.get('imgc', 3),
            task_num=task.get('task_num', 4),
            meta_decay=task.get('meta_decay', 1.),
            meta_decay_step_size=task.get('meta_decay_step_size', 2500),
            meta_lr=task.get('meta_lr', 1e-3),
            update_lr=task.get('update_lr', 1e-2),
            update_step=task.get('update_step', 5),
            update_step_test=task.get('update_step_test', 10),
            save_each_epoch=task.get('save_each_epoch', False),
            save_threshold=task.get('save_threshold', 0),
            test_each_epoch=task.get('test_each_epoch', False),
            log_print=task.get('log_print', False)
        )
