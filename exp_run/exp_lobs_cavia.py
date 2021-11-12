import sys

sys.path.append('..')

from methods.lobs_cavia import exp
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# gpu 0
# os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4eec6600-f5e3-f385-9b14-850ae9a2b236'
# gpu 1
os.environ["CUDA_VISIBLE_DEVICES"] = 'GPU-4b0856cd-c698-63a2-0b6e-9a33d380f9c4'

if __name__ == '__main__':

    tasks_train_net = [
        # cavia conv32_br mini-imagenet
        {
            'n_way': 5,
            'k_spt': 1,

            'dataset': 'mini-imagenet',
            'backbone': 'conv32_br',
            'task': 'train_net'
        },
        # cavia conv32_br cub
        {
            'n_way': 5,
            'k_spt': 1,

            'dataset': 'cub',
            'backbone': 'conv32_br',
            'task': 'train_net'
        }
    ]

    tasks_lobs = [
        # cavia conv32_br
        {
            'n_way': 5,
            'k_spt': 1,

            'dataset': 'mini-imagenet',
            'backbone': 'conv32_br',
            'task': 'lobs',
            'path_model': '/local/home/david/Remote/VibNet_Pytorch/model/lobs/mini-imagenet_conv32_br/net1-ori_[32,36,40,49,54].pkl'
        },
        {
            'n_way': 5,
            'k_spt': 1,

            'dataset': 'cub',
            'backbone': 'conv32_br',
            'task': 'lobs',
            'path_model': '/itet-stor/dawguo/net_scratch/ANP/model/lobs/cub_conv32_br/net2/net1-ori_[34,42,52,78,81].pkl'
        }
    ]

    for task in [tasks_train_net[1]]:
        exp(
            n_way=task['n_way'],
            k_spt=task['k_spt'],
            dataset=task['dataset'],
            backbone=task['backbone'],
            task=task['task'],
            path_model=task.get('path_model', None),

            k_qry=task.get('k_qry', 15),
            cr_steps=task.get('cr_steps', [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]),
            epoch_test=task.get('epoch_test', 5),
            epoch_retrain=task.get('epoch_retrain', 20),
            epoch=task.get('epoch', 60000),
            imgsz=task.get('imgsz', 84),
            imgc=task.get('imgc', 3),
            task_num=task.get('task_num', 4),
            meta_lr=task.get('meta_lr', 1e-3),
            update_lr=task.get('update_lr', 0.01),
            update_step=task.get('update_step', 5),
            update_step_test=task.get('update_step_test', 10),
            log_print=task.get('log_print', False)
        )
