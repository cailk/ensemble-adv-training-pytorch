# --coding:utf-8--
'''
@author: cailikun
@time: 2019/4/5 下午11:20
'''
import torch
import torchvision
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from mnist import *
from utils import train, test_error_rate
from attack_utils import gen_grad
from fgs import symbolic_fgs, iter_fgs
from os.path import basename
import argparse



def main(args):
    def get_model_type(model_name):
        model_type = {
            'models/modelA':0, 'models/modelA_adv':0, 'models/modelA_ens':0,
            'models/modelB':1, 'models/modelB_adv':1, 'models/modelB_ens':1,
            'models/modelC':2, 'models/modelC_adv':2, 'models/modelC_ens':2,
            'models/modelD':3, 'models/modelD_adv':3, 'models/modelD_ens':3,
        }
        if model_name not in model_type.keys():
            raise ValueError('Unknown model: {}'.format(model_name))
        return model_type[model_name]

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.cuda else 'cpu')

    '''
    Preprocess MNIST dataset
    '''
    kwargs = {'num_workers': 20, 'pin_memory': True} if args.cuda else {}
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../attack_mnist', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    src_model_name = args.src_model
    type = get_model_type(src_model_name)
    src_model = load_model(src_model_name, type).to(device)

    target_model_names = args.target_models
    target_models = [None] * len(target_model_names)
    for i in range(len(target_model_names)):
        type = get_model_type(target_model_names[i])
        target_models[i] = load_model(target_model_names[i], type=type).to(device)

    attack = args.attack
    if attack == 'test':
        correct_s = 0
        with torch.no_grad():
            for (data, labels) in test_loader:
                data, labels = data.to(device), labels.to(device)
                correct_s += test_error_rate(src_model, data, labels)
        err = 100. - 100. * correct_s / len(test_loader.dataset)
        print('Test error of {}: {:.2f}'.format(basename(src_model_name), err))

        for (name, target_model) in zip(target_model_names, target_models):
            correct_t = 0
            with torch.no_grad():
                for (data, labels) in test_loader:
                    data, labels = data.to(device), labels.to(device)
                    correct_t += test_error_rate(target_model, data, labels)
            err = 100. - 100. * correct_t / len(test_loader.dataset)
            print('Test error of {}: {:.2f}'.format(basename(target_model_names), err))
        return

    eps = args.eps

    correct = 0
    for (data, labels) in test_loader:
        if attack == 'rand_fgs':
            data = torch.clamp(data + torch.zeros_like(data).uniform_(-args.alpha, args.alpha), 0.0, 1.0)
            eps -= args.alpha
        data, labels = data.to(device), labels.to(device)
        grad = gen_grad(data, src_model, labels)

        if attack in ['fgs', 'rand_fgs']:
            adv_x = symbolic_fgs(data, grad, eps=eps)

        if attack == 'ifgs':
            adv_x = iter_fgs(src_model, data, labels, steps=args.steps, eps=args.eps/args.steps)

        correct += test_error_rate(src_model, adv_x, labels)
    test_error = 100. - 100. * correct / len(test_loader.dataset)
    print('Test Set Error Rate: {:.2f}%'.format(test_error))


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
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    main(args)
