# --coding:utf-8--
'''
@author: cailikun
@time: 2019/4/4 上午12:10
'''
import torch
from attack_utils import gen_grad

def symbolic_fgs(data, grad, eps=0.3, clipping=True):
    '''
    FGSM attack.
    '''
    # signed gradien
    normed_grad = grad.sign()

    # Multiply by constant epsilon
    scaled_grad = eps * normed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = data + scaled_grad
    if clipping:
        adv_x = torch.clamp(adv_x, 0, 1)
    return adv_x