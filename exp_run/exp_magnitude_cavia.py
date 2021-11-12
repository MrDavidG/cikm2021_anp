import sys

sys.path.append('..')

from methods.magnitude_cavia import exp
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == '__main__':
    tasks_magnitude_cub = [
        {
            'n_way': 5,
            'k_spt': 1,

            'dataset': 'cub',
            'backbone': 'conv32_br',
            'task': 'magnitude',

            'path_model': "/local/home/david/Remote/VibNet_Pytorch/model/lobs/cub_conv32_br/net1-ori_[10,19,27,36,91].pkl"
        }
    ]

    for task in tasks_magnitude_cub:
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
