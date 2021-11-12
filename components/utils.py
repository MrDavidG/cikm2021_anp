import torch
import numpy as np
from numpy.linalg import inv, pinv, LinAlgError


def calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def apply_grad(model, grad):
    '''
    assign gradient to model(nn.Module) instance. return the norm of gradient
    '''
    grad_norm = 0
    for p, g in zip(model.parameters(), grad):
        if p.grad is None:
            p.grad = g
        else:
            p.grad += g
        grad_norm += torch.sum(g ** 2)
    grad_norm = grad_norm ** (1 / 2)
    return grad_norm.item()


def mix_grad(grad_list, weight_list):
    '''
    calc weighted average of gradient
    '''
    mixed_grad = []
    for g_list in zip(*grad_list):
        g_list = torch.stack([weight_list[i] * g_list[i] for i in range(len(weight_list))])
        mixed_grad.append(torch.sum(g_list, dim=0))
    return mixed_grad


def get_accuracy(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points
    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(num_examples,)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())


def test_masks(masks, state_dict):
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


def mask_grad(grad, masks_dict):
    if masks_dict is None:
        return grad

    if 'vars.0' not in masks_dict.keys() and 'vars.2' not in masks_dict.keys():
        print('Keys in masks_dict is wrong!')
        print(masks_dict.keys())
        raise RuntimeError

    if type(grad) is tuple:
        grad = list(grad)

    for idx, v in enumerate(grad):
        k = 'vars.%d' % idx
        if k in masks_dict.keys():
            grad[idx] = v * torch.tensor(masks_dict[k], dtype=torch.float32).cuda()
    return grad


def unfold(kernel):
    k_shape = kernel.shape
    weight = np.zeros([k_shape[1] * k_shape[2] * k_shape[3], k_shape[0]])
    for i in range(k_shape[0]):
        weight[:, i] = np.reshape(kernel[i, :, :, :], [-1])

    return weight


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


def get_inv(mat):
    try:
        mat_inv = inv(mat)
    except LinAlgError:
        print(LinAlgError)
        mat_inv = pinv(mat)
    return mat_inv
