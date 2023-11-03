#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import matplotlib
matplotlib.use('Agg')
from torchvision import datasets, transforms
import torch
import copy

from utils.sampling import mnist_iid, mnist_noniid, emnist_iid, emnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNEmnist, Res_Model
from models.Fed import FedAvg
from models.test import test_img
from utils.__init__ import setup_seed


max_rounds_appr = 31000
global_round = 0
total_lines = 0


class Sat_Train_Info:
    def __init__(self):
        self.sat_id = 0
        self.model_round_id = 0


def read_satellites_order(filename, clients):
    global total_lines
    try:
        #打开文件
        fp = open(filename,"r")
        for line in fp.readlines():            
            line = line.replace('\n','')
            slices = line.split(' ')
            #以逗号为分隔符把数据转化为列表
            sat_id = int(slices[0])
            round = int(slices[1])              
            clients[total_lines].sat_id = sat_id
            clients[total_lines].model_round_id = round
            total_lines += 1          
        fp.close()
    except IOError:
        print("file open error, %s is not existing" % filename)


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    total_satellites = args.num_users
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
        net_glob = Res_Model().to(args.device)
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
    
    for iter in range(total_lines + 1):
        if os.path.exists('./save/model_w'+str(iter)+'.pth'): 
            os.remove('./save/model_w'+str(iter)+'.pth')

    # print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    torch.save(w_glob, './save/model_w'+str(global_round)+'.pth')

    clients = [Sat_Train_Info() for i in range(max_rounds_appr)]
    read_satellites_order("./output_info/satellites_order.txt", clients)
    clients = clients[:total_lines]
   

    for idx, client in enumerate(clients):      
        own_net = copy.deepcopy(net_glob)
        if os.path.exists('./save/model_w'+str(client.model_round_id)+'.pth'):
            own_network = torch.load('./save/model_w'+str(client.model_round_id)+'.pth')                
            own_net.load_state_dict(own_network)
    
        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[client.sat_id])
        w, loss = local.train(net=copy.deepcopy(own_net).to(args.device))
            
        w_base = own_net.state_dict()
        w_glob = FedAvg(w_glob, w_base, w, 1.0 / total_satellites)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        global_round += 1
        torch.save(net_glob.state_dict(), './save/model_w'+str(global_round)+'.pth')

        if idx % 50 == 0:
            net_glob.eval()
            # acc_train, _ = test_img(net_glob, dataset_train, args)
            acc_test, _ = test_img(net_glob, dataset_test, args)
            # print("Training accuracy: {:.2f}".format(acc_train))
            print("Testing accuracy: {:.2f}".format(acc_test))

