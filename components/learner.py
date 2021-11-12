import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class Learner(nn.Module):
    """

    """

    def __init__(self, config, imgc, imgsz):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()

        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        self.layers_name = list()
        self.layers_input = None

        self.kernel_size = None

        self.weights_map = dict()

        idx_layer = 1
        idx = 0
        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # w [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                torch.nn.init.kaiming_normal_(w)
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

            elif name is 'linear':
                # w [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
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
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # b [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                layer_name = 'cds%d' % idx_layer
                self.layers_name.append(layer_name)
                idx_layer += 1

                self.weights_map[layer_name] = ['vars.%d' % idx, 'vars.%d' % (idx + 1)]

                idx += 2

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d', 'identity_out', 'flatten', 'reshape',
                          'leakyrelu', 'sigmoid']:
                continue
            else:
                print(name)
                raise NotImplementedError

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)' % (param[0])
                info += tmp + '\n'


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info

    def forward(self, x, vars=None, bn_training=True):
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

        if vars is None:
            vars = self.vars

        idx = 0
        idx_layer = 0
        bn_idx = 0

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
