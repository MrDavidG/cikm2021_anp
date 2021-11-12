from torch.nn import Module
from torch.nn import ParameterList
from components.utils import calculate_fan_in_and_fan_out

import torch.nn.functional as F
import torch.nn as nn

import torch
import math


class Learner(Module):
    def __init__(self, config, num_context_params, context_in, register_context_params=False):
        super(Learner, self).__init__()

        self.config = config

        self.vars = ParameterList()
        self.vars_bn = ParameterList()

        self.layers_name = list()
        self.layers_input = None
        self.kernel_size = None
        self.weights_map = dict()

        self.num_context_params = num_context_params
        self.context_in = context_in

        idx_layer = 1
        idx = 0
        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # w [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                torch.nn.init.kaiming_uniform_(w, nonlinearity='relu')
                self.vars.append(w)
                # b [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # layer_name
                layer_name = 'c%d' % idx_layer
                self.layers_name.append(layer_name)
                idx_layer += 1

                # map
                self.weights_map[layer_name] = ['vars.%d' % idx, 'vars.%d' % (idx + 1)]
                idx += 2

                if self.kernel_size is None:
                    self.kernel_size = param[2]

                n_filter = param[0]

            elif name is 'linear':
                # w [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_uniform_(w, nonlinearity='linear')
                self.vars.append(w)
                # b [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # layer_name
                layer_name = 'f%d' % idx_layer
                self.layers_name.append(layer_name)
                idx_layer += 1

                # map
                self.weights_map[layer_name] = ['vars.%d' % idx, 'vars.%d' % (idx + 1)]
                idx += 2

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                idx += 2

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])

            elif name is 'identity_in':
                # w [ch_out, ch_in, 1, 1]
                w = nn.Parameter(torch.ones(*param[:4]))
                torch.nn.init.kaiming_uniform_(w, nonlinearity='relu')
                self.vars.append(w)
                # b [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                layer_name = 'cds%d' % idx_layer
                self.layers_name.append(layer_name)
                idx_layer += 1

                self.weights_map[layer_name] = ['vars.%d' % idx, 'vars.%d' % (idx + 1)]

                idx += 2

            elif name is 'max_pool2d':
                if self.context_in[idx_layer - 2]:
                    # TODO: 在这里初始化所有的film层权重
                    w_film = nn.Parameter(torch.ones(2 * n_filter, self.num_context_params))
                    torch.nn.init.kaiming_uniform_(w_film, nonlinearity='relu')
                    self.vars.append(w_film)
                    # 这里不能以0为初始化
                    b_film = nn.Parameter(torch.zeros(2 * n_filter))
                    fan_in, _ = calculate_fan_in_and_fan_out(w_film)
                    bound = 1 / math.sqrt(fan_in)
                    torch.nn.init.uniform_(b_film, -bound, bound)
                    self.vars.append(b_film)

                    idx += 2

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'identity_out', 'flatten', 'reshape',
                          'leakyrelu', 'sigmoid']:
                continue
            else:
                print(name)
                raise NotImplementedError

        if register_context_params:
            self.context_params = nn.Parameter(torch.zeros(size=[self.num_context_params]))
            self.vars.append(self.context_params)
        else:
            self.context_params = torch.zeros(size=[self.num_context_params], requires_grad=True).cuda()

    def forward(self, x, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        idx = 0
        idx_layer = 0
        bn_idx = 0

        vars = self.vars

        self.layers_input = dict()

        for name, param in self.config:
            if name is 'conv2d':
                self.layers_input[self.layers_name[idx_layer]] = x
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                idx_layer += 1
            elif name is 'identity_in':
                self.layers_input[self.layers_name[idx_layer]] = x
                w, b = vars[idx], vars[idx + 1]
                identity = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                idx_layer += 1
            elif name is 'identity_out':
                x = x + identity
                identity = None
            elif name is 'linear':
                self.layers_input[self.layers_name[idx_layer]] = x
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                idx_layer += 1
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            elif name is 'flatten':
                x = x.view(x.size(0), -1)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, inplace=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])

                # film layer
                if self.context_in[idx_layer - 1]:
                    w, b = vars[idx], vars[idx + 1]
                    film = F.linear(self.context_params, w, b)
                    gamma = film[:int(film.size(0) / 2)].view(1, -1, 1, 1)
                    beta = film[int(film.size(0) / 2):].view(1, -1, 1, 1)
                    x = gamma * x + beta
                    idx += 2
            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return x

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars

    def reset_context_params(self):
        self.context_params = self.context_params.detach() * 0
        self.context_params.requires_grad = True
