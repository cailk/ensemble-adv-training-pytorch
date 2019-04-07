# --coding:utf-8--
'''
@author: cailikun
@time: 2019/4/5 下午11:20
'''
import torch
from mnist import load_model
from utils import test_error_rate
from os.path import basename
import numpy as np
import argparse


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

    src_model_name = args.src_model
    src_model = load_model(src_model_name).to(device)

    target_model_names = args.target_models
    target_models = [None] * len(target_model_names)
    for i in range(len(target_model_names)):
        type = get_model_type(target_model_names[i])
        target_models[i] = load_model(target_model_names[i], type=type).to(device)

    if attack == 'test':
        err = test_error_rate(src_model, test_loader, cuda=args.cuda)
        print('Test error of {}: {:.2f}'.format(basename(src_model_name), err))

        for (name, target_model) in zip(target_model_names, target_models):
            err = test_error_rate(target_model, test_loader, cuda=args.cuda)
            print('Test error of {}: {:.2f}'.format(basename(target_model_names), err))
        return

    eps = args.eps









if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple eval')
    parser.add_argument('attack', choices=['test', 'fgs', 'ifgs', 'rand_fgs', 'CW'], help='Name of attack')
    parser.add_argument('src_model', help='Source model for attack')
    parser.add_argument('target_models', nargs='*', help='path to target model(s)')
    parser.add_argument('--batch_size', type=int, default=64, help='Size of training batches (default: 64)')
    parser.add_argument('--eps', type=float, default=0.3, help='FGS attack scale (default: 0.3)')
    parser.add_argument('--alpha', type=float, default=0.05, help='RAND+FGSM random pertubation scale')
    parser.add_argument('--steps', type=int, default=10, help='Iterated FGS steps (default: 10)')
    parser.add_argument('--kappa', type=float, default=100, help='CW attack confidence')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--disable_cuda', action='store_true', default=False, help='Disable CUDA (default: False)')


    args = parser.parse_args()
