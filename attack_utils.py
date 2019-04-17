# --coding:utf-8--
'''
@author: cailikun
@time: 19-3-27 下午7:07
'''

import torch
import torch.nn.functional as F

def gen_adv_loss(logits, labels, loss='logloss', mean=False):
    '''
    Generate the loss function
    '''
    if loss == 'training':
        # use the model's output instead of the true labels to avoid
        # label leaking at training time
        labels = logits.max(1)[1]
        if mean:
            out = F.cross_entropy(logits, labels, reduction='mean')
        else:
            out = F.cross_entropy(logits, labels, reduction='sum')
    elif loss == 'logloss':
        if mean:
            out = F.cross_entropy(logits, labels, reduction='mean')
        else:
            out = F.cross_entropy(logits, labels, reduction='sum')
    else:
        raise ValueError('Unknown loss: {}'.format(loss))
    return out

def gen_grad(x, model, y, loss='logloss'):
    '''
    Generate the gradient of the loss function.
    '''
    model.eval()
    x.requires_grad = True

    # Define gradient of loss wrt input
    logits = model(x)
    adv_loss = gen_adv_loss(logits, y, loss)
    model.zero_grad()
    adv_loss.backward()
    grad = x.grad.data
    return grad
