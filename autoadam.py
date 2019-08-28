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

import math
import torch
from autooptimizer import AutoOptimizer


class AutoAdam(AutoOptimizer):
    """Implements AutoAdam algorithm.
    Arguments:
        model (torch.nn.Module): Model containing the parameters to optimize
        beta2 (float, optional): coefficient used for computing
            running averages of gradient and its square (default: 0.999)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    """

    def __init__(self, model, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, ewma=0.9, gamma0=0.999):
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, ewma=ewma, gamma0=gamma0)
        super(AutoAdam, self).__init__(model, defaults)

    def __setstate__(self, state):
        super(AutoAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None, verbose=False):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            verbose: Be verbose.
        """
        super(AutoAdam, self).step(closure=closure)

        loss = None
        if closure is not None:
            loss = closure()

        self.model.auto_params = {'lr': [], 'momentum': []}

        for group in self.param_groups:
            for param in group['params']:

                if param.grad is None:
                    continue

                if torch.sum(torch.abs(param.grad)) == 0:
                    continue

                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AutoAdam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[param]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(param.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(param.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(param.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad_all = param.grad_all.data
                    grad_all.add_(group['weight_decay'], param.data)
                    grad.add_(group['weight_decay'], param.data)

                # Decay the first and second moment running average coefficient
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                sqrt_bias_correction2 = math.sqrt(1 - beta2 ** state['step']) / (1 - beta1)
                hessian = denom / sqrt_bias_correction2
                # print(torch.norm(hessian))
                self.auto_tune(parameter=param, hessian=hessian, verbose=verbose)
                group['lr'] = 1 - param.gamma[0]
                adaptive_beta1 = param.gamma[1] / (1 - param.gamma[0])

                self.model.auto_params['lr'].append(group['lr'].item())
                self.model.auto_params['momentum'].append(adaptive_beta1.item())

                exp_avg.mul_(adaptive_beta1).add_(1 - adaptive_beta1, grad)

                step_size = group['lr'] * sqrt_bias_correction2

                param.data.addcdiv_(-step_size, exp_avg, denom)

        return loss