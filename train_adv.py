# --coding:utf-8--
'''
@author: cailikun
@time: 2019/4/2 下午2:13
'''

import torch
import torchvision
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from mnist import *
from utils import train, test_error_rate
from attack_utils import gen_grad
from fgs import symbolic_fgs
import argparse
import os

def main(args):
    def get_model_type(adv_model_name):
        if adv_model_name == 'models/modelA':
            type = 0
        elif adv_model_name == 'models/modelB':
            type = 1
        elif adv_model_name == 'models/modelC':
            type = 2
        elif adv_model_name == 'models/modelD':
            type = 3
        else:
            raise ValueError('Unknown model: {}'.format(adv_model_name))
        return type

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.cuda else 'cpu')

    '''
    Preprocess MNIST dataset
    '''
    kwargs = {'num_workers': 20, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../attack_mnist', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../attack_mnist', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    eps = args.eps
    adv_model_names = args.adv_models
    adv_models = [None] * len(adv_model_names)
    for i in range(len(adv_model_names)):
        type = get_model_type(adv_model_names[i])
        adv_models[i] = load_model(adv_model_names[i], type=type).to(device)

    model = model_mnist(type=args.type).to(device)
    optimizer = optim.Adam(model.parameters())

    x_advs = [None] * (len(adv_models) + 1)
    for epoch in range(args.epochs):
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            for i, m in enumerate(adv_models + [model]):
                grad = gen_grad(data, m, labels, loss='training')
                x_advs[i] = symbolic_fgs(data, grad, eps=eps)
            train(epoch, batch_idx, model, data, labels, optimizer, x_advs=x_advs)
    test_error = test_error_rate(model, test_loader, cuda=args.cuda)
    print('Test Set Error Rate: {:.2f}%'.format(test_error))

    torch.save(model.state_dict(), args.model + '.pkl')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial Training MNIST model')
    parser.add_argument('model', help='path to model')
    parser.add_argument('adv_models', nargs='*', help='path to adv model(s)')
    parser.add_argument('--type', type=int, default=0, help='Model type (default: 0)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--disable_cuda', action='store_true', default=False, help='Disable CUDA (default: False)')
    parser.add_argument('--batch_size', type=int, default=64, help='Size of training batches (default: 64)')
    parser.add_argument('--epochs', type=int, default=12, help='Number of epochs (default: 12)')
    parser.add_argument('--eps', type=float, default=0.3, help='FGSM attack scale (default: 0.3)')

    args = parser.parse_args()
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    main(args)
