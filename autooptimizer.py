""" 
Copyright 2019 eBay Inc.
Developers/Architects: Selcuk Kopru, Tomer Lancewicki
 
Licensed under the Apache License, Version 2.0 (the "License");
You may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
import numpy as np
import torch
from torch.optim import Optimizer


gradients = {}
conv_names = ['dA', 'dW', 'db']
fc_names = ['db', 'dA', 'dW']

GAMMA0_DEFAULT = 0.999
EWMA_DEFAULT = 0.9


def store_gradients(name, self, grad_input, grad_output):
    """
    Backward hook function used to store individual gradients.
    This function is used as a parameter for register_backward_hook() in PyTorch.
    For further information, see https://pytorch.org/docs/stable/nn.html#torch.nn.Module.register_backward_hook
    :param name: Layer name
    :param self: Self instance
    :param grad_input: Input gradient
    :param grad_output: Output gradient
    :return:
    """
    for index, grad in enumerate(grad_input):
        if name.startswith('conv'):
            gradients[(name, conv_names[index])] = grad
        else:
            gradients[(name, fc_names[index])] = grad
    gradients[(name, 'dZ')] = grad_output[0]


class AutoOptimizer(Optimizer):
    """
    Base class for all auto optimizers. Provides basic functionality such as individual gradient computation,
    variance estimation and automated tuning that is used in derived optimizers.
    """

    def __init__(self, model, defaults):
        if 'gamma0' not in defaults:
            defaults['gamma0'] = GAMMA0_DEFAULT
        if 'ewma' not in defaults:
            defaults['ewma'] = EWMA_DEFAULT

        self.model = model
        self._prep_model(defaults['gamma0'])
        self.N = 0
        self.f_w_x = None
        self.ewma = defaults['ewma']
        super(AutoOptimizer, self).__init__(self.model.parameters(), defaults)

    def __setstate__(self, state):
        pass

    def _prep_model(self, gamma0=0.999):
        """
        Prepare the model for auto optimization. This includes registering the backward hook and initializing
        variables.
        :param gamma0: Initial value for gamma0.
        """
        input_layer = True
        for name, layer in self.model._modules.items():
            if hasattr(layer, 'weight'):
                layer.dZ = None

                if input_layer:
                    layer.input_layer = True
                    input_layer = False
                else:
                    layer.input_layer = False

                layer.register_backward_hook(functools.partial(store_gradients, name))
                for parameter in [layer.weight, layer.bias]:
                    parameter.layer_name = name
                    parameter.gradient_est = torch.zeros(parameter.shape, device=parameter.device)
                    parameter.gamma = torch.tensor([gamma0, 0.0], device=parameter.device)

    def step(self, closure=None):
        """
        This function should be called from all step functions in derived optimizers.
        Set variables and compute individual gradients.
        """
        self.f_w_x = self.model.loss_all
        self.N = self.f_w_x.shape[0]
        self.compute_individual_gradients(self.N)

    def compute_individual_gradients(self, N):
        """
        Compute individual gradients for each layer in the model.
        :param N: Batch size
        :return:
        """
        for name, layer in self.model._modules.items():
            if isinstance(layer, torch.nn.Conv2d):
                layer.dZ = gradients[(name, 'dZ')]
            elif isinstance(layer, torch.nn.Linear):
                layer.dZ = gradients[(name, 'dZ')].t()
            else:
                continue

            if isinstance(layer, torch.nn.Linear):
                layer.weight.grad_all = torch.bmm(layer.dZ.t().unsqueeze(2), layer.A_prev.t().unsqueeze(1)) * N
                layer.bias.grad_all = gradients[(name, 'dZ')] * N
            elif isinstance(layer, torch.nn.Conv2d):
                layer.weight.grad_all = torch.zeros([N] + list(layer.weight.grad.shape))
                for n in range(N):
                    layer.weight.grad_all[n] = torch.nn.functional.conv2d(
                        layer.A_prev[n].unsqueeze(1), layer.dZ[n].unsqueeze(1)).transpose(1, 0) * N
                layer.bias.grad_all = gradients[(name, 'dZ')].sum((2, 3)) * N
            elif isinstance(layer, torch.nn.LSTM):
                layer.weight.grad_all = torch.zeros()

    def compute_var(self, parameter, hessian=None):
        """
        Compute variance estimate. If Hessian is None, it is assumed to be equal to the identity matrix, e.g. in SGD.
        :param parameter: Model parameter.
        :param hessian: Hessian matrix.
        :param N: Mini-batch size.
        :return:
        """
        if hessian is None:
            parameter.var_est = torch.pow(torch.add(parameter.grad_all, -parameter.grad), 2).sum().item() / \
                                (self.N * (self.N - 1))
        else:
            parameter.var_est = torch.div(torch.pow(torch.add(parameter.grad_all, -parameter.grad), 2),
                                          hessian).sum().item() / (self.N * (self.N - 1))

    def auto_tune(self, parameter, hessian=None, with_momentum=True, verbose=None):
        """
        Estimate the oracle vector gamma given the gradients for the mini-batch observations
        at that step and the gradient estimator computed at a previous step. For SGD, Hessian
        is assumed to be the identity matrix.
        :param parameter: Parameter to be auto-tuned.
        :param hessian: Hessian matrix
        :param with_momentum: True if the AutoOptimizer is calculating the momentum. It is
                  False otherwise (as in AutoAdaGrad).
        :param verbose: Be verbose and print computed values.
        """
        G = torch.stack((parameter.grad, parameter.grad - parameter.gradient_est))
        if hessian is None:
            B = G
        else:
            B = torch.div(G, hessian)

        A = torch.zeros([2, 2])
        A[0][0] = torch.sum((B[0] * G[0])).item()
        A[0][1] = A[1][0] = torch.sum((B[0] * G[1])).item()
        A[1][1] = torch.sum((B[1] * G[1])).item()

        self.compute_var(parameter=parameter, hessian=hessian)
        if verbose:
            print('N: ', self.N)
            print('layer_name: ', parameter.layer_name)
            print('parameter.grad.shape: ', parameter.grad.shape)
            print('parameter.gradient_est.shape: ', parameter.gradient_est.shape)
            print('sum(abs(parameter.grad)): ', torch.sum(torch.abs(parameter.grad)).item())
            print('G.shape: ', G.shape)
            print('parameter.var_est: ', parameter.var_est)
            print('A: ', A.numpy())

        if np.linalg.det(A.numpy()) == 0.0 or np.linalg.matrix_rank(A.numpy()) < 2:
            gamma = np.array([min(parameter.var_est / A[0, 0], 0.999), 0])
        else:
            gamma = torch.matmul(A.inverse(), parameter.var_est * torch.ones(2)).data.numpy()

        if verbose:
            print('(1) gamma: ', gamma)

        if not with_momentum or gamma[0] < 0 or gamma[0] > 0.999 or gamma[1] < 0 or gamma[1] >= (1 - gamma[0]) or np.isnan(gamma).any():
            gamma = np.array([min(parameter.var_est / A[0, 0], 0.999), 0])

        parameter.gamma = torch.tensor([
            min((1 - self.ewma) * gamma[0] + self.ewma * parameter.gamma[0], 0.999),
            min((1 - self.ewma) * gamma[1] + self.ewma * parameter.gamma[1], 0.999)
        ]).float()

        if verbose:
            print('(2) gamma: ', gamma)
            print('parameter.gamma: ', parameter.gamma)
            print('*' * 80)

        parameter.gradient_est = (1 - parameter.gamma[0] - parameter.gamma[1]) * parameter.grad + \
            parameter.gamma[1] * parameter.gradient_est