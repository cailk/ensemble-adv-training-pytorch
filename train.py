# --coding:utf-8--
'''
@author: cailikun
@time: 19-3-26 上午10:26
'''

import torch
import torchvision
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from mnist import *
from utils import train, test
import argparse
import os


def main(args):
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

    model = model_mnist(type=args.type).to(device)
    optimizer = optim.Adam(model.parameters())

    # Train an MNIST model
    for epoch in range(args.epochs):
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            train(epoch, batch_idx, model, data, labels, optimizer)

    # Finally print the result!
    correct = 0
    with torch.no_grad():
        for (data, labels) in test_loader:
            data, labels = data.to(device), labels.to(device)
            correct += test(model, data, labels)
    test_error = 100. - 100. * correct / len(test_loader.dataset)
    print('Test Set Error Rate: {:.2f}%'.format(test_error))

    torch.save(model.state_dict(), args.model+'.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training MNIST model')
    parser.add_argument('model', help='path to model')
    parser.add_argument('--type', type=int, default=1, help='Model type (default: 1)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--disable_cuda', action='store_true', default=False, help='Disable CUDA (default: False)')
    parser.add_argument('--batch_size', type=int, default=64, help='Size of training batches (default: 64)')
    parser.add_argument('--epochs', type=int, default=6, help='Number of epochs to train (default: 6)')
    #parser.print_help()
    args = parser.parse_args()
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    main(args)


