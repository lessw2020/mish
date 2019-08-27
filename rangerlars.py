#RangerLars / Over9000  -
#credit to Federico (https://gist.github.com/redknightlois/c4023d393eb8f92bb44b2ab582d7ec20) and
#@mgrankin  (https://github.com/mgrankin/over9000) for adding Lars into Ranger (Lookahead + RAdam)

#this version integrates several improvements from @oquiza and Yaroslav Geraskin, added by Federico.
#8/27/19

import torch, math
from torch.optim.optimizer import Optimizer
import itertools as it

class RangerLars(Optimizer):

    def __init__(self, params, lr=1e-3, alpha=.5, k= 5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}') 
            
        defaults = dict(lr=lr, alpha=alpha, k=k, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        
        super().__init__(params, defaults)
        
        # look ahead params
        for group in self.param_groups:
            group["step_counter"] = 0

        self.alpha = alpha
        self.k = k 

        #lookahead weights
        self.slow_weights = [[p.clone().detach() for p in group['params']]
                                for group in self.param_groups]
        
        #don't use grad for lookahead weights
        for w in it.chain(*self.slow_weights):
            w.requires_grad = False
        


    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):

        loss = None
        #if closure is not None:
        #    loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RangerLars does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, radam_step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        radam_step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        radam_step_size = 1 / (1 - beta1 ** state['step'])
                    buffered[2] = radam_step_size

                update = torch.zeros_like(p_data_fp32)
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    update.addcdiv_(radam_step_size, exp_avg, denom)
                else:
                    update.add_(radam_step_size, exp_avg)

                if group['weight_decay'] != 0:
                    update.add_(group['weight_decay'], p_data_fp32)

                radam_norm = update.pow(2).sum().sqrt()
                weight_norm = p.data.pow(2).sum().sqrt()
                if weight_norm == 0 or radam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / radam_norm

                trust_ratio = max(0, min(10, trust_ratio))

                state['weight_norm'] = weight_norm
                state['adam_norm'] = radam_norm
                state['trust_ratio'] = trust_ratio

                p_data_fp32.add_(-update * trust_ratio * group['lr'])
                p.data.copy_(p_data_fp32)

        
        #look ahead tracking and updating if latest batch = k
        for group,slow_weights in zip(self.param_groups,self.slow_weights):
            group['step_counter'] += 1
            if group['step_counter'] % self.k != 0:
                continue
            for p,q in zip(group['params'],slow_weights):
                if p.grad is None:
                    continue
                #at k interval: take the difference of (RAdam params - LookAhead params) * LookAhead alpha param
                q.data.add_(self.alpha,p.data - q.data) 
                #update RAdam weights with the interpolated weights
                p.data.copy_(q.data)        
            
        return loss        
