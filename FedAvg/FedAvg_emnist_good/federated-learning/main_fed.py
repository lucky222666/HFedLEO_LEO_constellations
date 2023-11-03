#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import math
import joblib

from utils.sampling import mnist_iid, mnist_noniid, emnist_iid, emnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNEmnist, Res_Model
from models.Fed import FedAvg
from models.test import test_img


fraction0 = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
clients = [
    [22, 27, 24, 23, 43, 28, 9, 42, 38, 8, 6, 63, 7, 64, 36, 39, 5, 46, 65, 11, 12, 16, 59, 62, 10, 25, 45, 19, 61, 21, 26, 31, 18, 35, 44, 41, 20, 40, 30, 15, 37, 33, 29, 13, 34, 32, 60, 47, 17, 66, 14, 48, 67, 49, 68, 50, 69, 51, 52, 70, 71, 53, 0, 54, 1, 55, 2, 56, 3, 4, 58, 57],
    [55, 52, 2, 39, 54, 20, 1, 53, 69, 4, 37, 22, 38, 21, 36, 23, 35, 41, 5, 42, 46, 0, 68, 57, 60, 40, 50, 3, 51, 56, 70, 65, 61, 19, 63, 48, 59, 45, 24, 49, 67, 43, 64, 58, 44, 18, 47, 6, 66, 62, 17, 7, 25, 8, 27, 10, 28, 11, 29, 12, 30, 13, 31, 14, 32],
    [4, 62, 57, 67, 66, 61, 30, 60, 31, 12, 46, 11, 29, 44, 47, 28, 48, 43, 49, 5, 42, 52, 53, 45, 3, 50, 63, 27, 71, 14, 58, 10, 9, 8, 59, 2, 6, 56, 65, 32, 7, 51, 70, 64, 54, 55, 26, 1, 69, 0, 68, 15, 33, 13, 16, 34, 17, 35],
    [7, 65, 60, 70, 66, 69, 33, 34, 15, 49, 14, 32, 50, 31, 51, 64, 46, 4, 8, 55, 45, 68, 63, 67, 10, 6, 48, 53, 61, 17, 13, 11, 30, 2, 12, 62, 9, 5, 0, 47, 59, 35, 16, 54, 36, 52, 71, 1, 57, 58, 29],
    [70, 65, 61, 8, 35, 69, 33, 48, 32, 50, 15, 47, 51, 31, 5, 52, 46, 4, 71, 56, 34, 49, 68, 16, 3, 64, 63, 11, 7, 60, 67, 10, 6, 13, 66, 9, 12, 14, 62, 1, 18, 55, 17, 53],
    [64, 69, 12, 8, 65, 14, 66, 46, 9, 51, 70, 61, 4, 33, 49, 48, 34, 50, 6, 67, 47, 16, 35, 53, 30, 54, 59, 58, 32, 13, 63, 62, 29, 15, 10, 11],
    [65, 13, 61, 70, 8, 71, 30, 35, 50, 69, 33, 48, 34, 32, 15, 51, 67, 47, 52, 5, 46, 0, 4, 57, 49, 68, 64, 11, 7],
    [70, 65, 7, 13, 66, 64, 16, 14, 62, 15, 10, 47, 33, 34, 5, 49, 50, 32, 51, 35, 31, 69],
    [64, 67, 66, 13, 9, 65, 51, 70, 12, 50, 49, 33, 34, 48, 7],
    [66, 61, 70, 13, 8, 65, 71, 67],
]


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'emnist':
        trans_emnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.EMNIST('../data/emnist/', split="bymerge", train=True, download=True, transform=trans_emnist)
        dataset_test = datasets.EMNIST('../data/emnist/', split="bymerge", train=False, download=True, transform=trans_emnist)
        # sample users
        if args.iid:
            dict_users = emnist_iid(dataset_train, args.num_users)
        else:
            dict_users = emnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
          
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = Res_Model(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'emnist':
        net_glob = CNNEmnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
        
    # network_state_dict = torch.load('model.pth')
    # net_glob.load_state_dict(network_state_dict)
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

 
    for iter in range(args.epochs):
        loss_locals = []
        idxs_users = []
        
        per_round_clients = clients[iter]   # iter可以理解为索引
        if fraction0[iter] == 1.0:
            w_locals = [w_glob for i in range(args.num_users)] 
        else:
            w_locals = []
        
        # 生成这一轮参与训练的客户端
        for i in per_round_clients:
            idxs_users.extend(range(i * 22, (i + 1) * 22))
        
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if fraction0[iter] == 1.0:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        joblib.dump(w_glob, './save/w_glob.pkl')
        joblib.dump(w_locals[0], './save/w_local.pkl')
        torch.save(w_glob, './save/w_glob1')
        torch.save(w_locals[0], './save/w_local1')
        
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # torch.save(net_glob.state_dict(), './model.pth')

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        
        
        # testing
        net_glob.eval()
        acc_train, loss_train2 = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))

