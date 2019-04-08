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
    normed_grad = grad.detach().sign()

    # Multiply by constant epsilon
    scaled_grad = eps * normed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = data.detach() + scaled_grad
    if clipping:
        adv_x = torch.clamp(adv_x, 0, 1)
    return adv_x

def iter_fgs(model, data, labels, steps, eps):
    '''
    I-FGSM attack.
    '''
    adv_x = data

    # iteratively apply the FGSM with small step size
    for i in range(steps):
        grad = gen_grad(adv_x, model, labels)
        adv_x = symbolic_fgs(adv_x, grad, eps)
    return adv_x