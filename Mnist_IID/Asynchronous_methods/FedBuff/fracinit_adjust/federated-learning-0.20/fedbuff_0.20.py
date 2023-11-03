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
from utils.__init__ import *


fraction20 = [0.20] * 30
clients = [
    [20, 21, 22, 23, 27, 28, 29, 30, 31, 32, 37, 38, 39, 40, 41, 42, 46, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 66, 67, 68, 69, 70, 71, 75, 76, 77, 78, 79],
    [30, 31, 32, 34, 35, 36, 40, 41, 42, 43, 44, 45, 46, 50, 51, 52, 53, 54, 55, 59, 60, 61, 63, 64, 65, 69, 70, 71, 72, 73, 74, 75, 78, 79, 80, 81, 82, 83, 88, 89],
    [43, 44, 45, 46, 47, 48, 49, 53, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 72, 73, 74, 75, 76, 77, 78, 82, 83, 84, 85, 86, 87, 88, 91, 92, 93, 94, 95],
    [42, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 66, 67, 68, 69, 70, 71, 72, 76, 77, 78, 79, 80, 81, 85, 86, 87, 88, 89, 90, 91, 95, 96, 97, 98, 99, 105, 106, 107, 108],
    [54, 55, 56, 60, 61, 62, 63, 64, 65, 66, 70, 71, 72, 73, 74, 75, 79, 80, 81, 82, 83, 84, 85, 89, 90, 91, 92, 93, 94, 99, 100, 101, 102, 103, 104, 108, 109, 110, 118, 119],
    [67, 68, 69, 73, 74, 75, 76, 77, 78, 79, 83, 84, 85, 86, 87, 88, 89, 92, 93, 94, 95, 96, 97, 98, 102, 103, 104, 105, 106, 107, 108, 112, 113, 114, 115, 116, 117, 121, 122, 123],
    [128, 129, 135, 70, 71, 72, 79, 80, 81, 82, 86, 87, 88, 89, 90, 91, 92, 96, 97, 98, 99, 100, 101, 105, 106, 107, 108, 109, 110, 111, 115, 116, 117, 118, 119, 120, 121, 125, 126, 127],
    [128, 129, 130, 131, 132, 133, 134, 138, 139, 148, 82, 83, 84, 85, 90, 91, 92, 93, 94, 95, 99, 100, 101, 102, 103, 104, 105, 109, 110, 111, 112, 113, 114, 118, 119, 120, 121, 122, 123, 124],
    [131, 132, 133, 134, 135, 136, 137, 141, 142, 143, 144, 145, 146, 147, 94, 95, 96, 97, 98, 102, 103, 104, 105, 106, 107, 108, 112, 113, 114, 115, 116, 117, 118, 121, 122, 123, 124, 125, 126, 127],
    [128, 129, 130, 135, 136, 137, 138, 139, 140, 144, 145, 146, 147, 148, 149, 150, 154, 155, 156, 157, 158, 159, 100, 101, 106, 107, 108, 109, 110, 111, 115, 116, 117, 118, 119, 120, 121, 125, 126, 127],
    [129, 130, 131, 132, 133, 134, 139, 140, 141, 142, 143, 144, 148, 149, 150, 151, 152, 153, 154, 158, 159, 160, 161, 162, 167, 168, 169, 110, 111, 112, 113, 114, 115, 119, 120, 121, 122, 123, 124, 125],
    [128, 132, 133, 134, 135, 136, 137, 138, 142, 143, 144, 145, 146, 147, 151, 152, 153, 154, 155, 156, 157, 161, 162, 163, 164, 165, 166, 167, 170, 171, 172, 173, 174, 175, 122, 123, 124, 125, 126, 127],
    [130, 131, 135, 136, 137, 138, 139, 140, 141, 145, 146, 147, 148, 149, 150, 155, 156, 157, 158, 159, 160, 164, 165, 166, 167, 168, 169, 170, 174, 175, 176, 177, 178, 179, 183, 184, 185, 186, 187, 121],
    [133, 134, 140, 141, 142, 143, 144, 148, 149, 150, 151, 152, 153, 154, 158, 159, 160, 161, 162, 163, 168, 169, 170, 171, 172, 173, 177, 178, 179, 180, 181, 182, 183, 187, 188, 189, 196, 197, 198, 199],
    [0, 1, 2, 9, 146, 147, 151, 152, 153, 154, 155, 156, 157, 161, 162, 163, 164, 165, 166, 167, 170, 171, 172, 173, 174, 175, 176, 180, 181, 182, 183, 184, 185, 186, 191, 192, 193, 194, 195, 196],
    [3, 4, 5, 6, 7, 8, 9, 13, 14, 150, 158, 159, 160, 164, 165, 166, 167, 168, 169, 170, 174, 175, 176, 177, 178, 179, 183, 184, 185, 186, 187, 188, 189, 193, 194, 195, 196, 197, 198, 199],
    [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 26, 27, 161, 162, 163, 170, 171, 172, 173, 177, 178, 179, 180, 181, 182, 183, 187, 188, 189, 190, 191, 192, 196, 197, 198, 199],
    [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 24, 25, 29, 39, 173, 174, 175, 176, 180, 181, 182, 183, 184, 185, 186, 190, 191, 192, 193, 194, 195, 196],
    [3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36, 37, 38, 185, 186, 187, 188, 189, 193, 194, 195, 196, 197, 198, 199],
    [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 35, 36, 37, 38, 39, 40, 41, 45, 46, 47, 48, 49, 190, 191, 192, 198, 199],
    [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 29, 30, 31, 32, 33, 34, 39, 40, 41, 42, 43, 44, 48, 49, 50, 51, 52, 53, 54, 58, 59],
    [13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 33, 34, 35, 36, 37, 38, 42, 43, 44, 45, 46, 47, 48, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66],
    [20, 21, 22, 26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 46, 47, 48, 49, 50, 51, 55, 56, 57, 58, 59, 60, 61, 65, 66, 67, 68, 69, 74, 75, 76, 77, 78, 79],
    [25, 30, 31, 32, 33, 34, 35, 39, 40, 41, 42, 43, 44, 45, 49, 50, 51, 52, 53, 54, 58, 59, 60, 61, 62, 63, 64, 68, 69, 70, 71, 72, 73, 74, 78, 79, 80, 87, 88, 89],
    [38, 42, 43, 44, 45, 46, 47, 48, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 67, 71, 72, 73, 74, 75, 76, 77, 81, 82, 83, 84, 85, 86, 87, 90, 91, 92, 93],
    [40, 41, 50, 51, 55, 56, 57, 58, 59, 60, 61, 65, 66, 67, 68, 69, 70, 74, 75, 76, 77, 78, 79, 80, 84, 85, 86, 87, 88, 89, 90, 94, 95, 96, 97, 98, 99, 104, 105, 106],
    [52, 53, 54, 60, 61, 62, 63, 64, 68, 69, 70, 71, 72, 73, 74, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92, 93, 97, 98, 99, 100, 101, 102, 103, 107, 108, 109, 117, 118],
    [65, 66, 67, 71, 72, 73, 74, 75, 76, 77, 81, 82, 83, 84, 85, 86, 87, 90, 91, 92, 93, 94, 95, 96, 100, 101, 102, 103, 104, 105, 106, 110, 111, 112, 113, 114, 115, 116, 120, 121],
    [128, 129, 133, 70, 77, 78, 79, 80, 84, 85, 86, 87, 88, 89, 90, 94, 95, 96, 97, 98, 99, 103, 104, 105, 106, 107, 108, 109, 113, 114, 115, 116, 117, 118, 119, 123, 124, 125, 126, 127],
    [128, 129, 130, 131, 132, 136, 137, 138, 139, 145, 80, 81, 82, 83, 90, 91, 92, 93, 97, 98, 99, 100, 101, 102, 103, 107, 108, 109, 110, 111, 112, 116, 117, 118, 119, 120, 121, 122, 126, 127],
]


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 在分数据集之前 先设计一个比较友好的随机种子
    setup_seed(20)

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
        if fraction20[iter] == 1.0:
            w_locals = [w_glob for i in range(args.num_users)] 
        else:
            w_locals = []
        
        # 生成这一轮参与训练的客户端
        for i in per_round_clients:
            idxs_users.append(i)
        
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if fraction20[iter] == 1.0:
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

