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
import os
import random

from utils.sampling import iid, noniid
from utils.options import args_parser
from models.Update import LocalUpdate, LocalTest
from models.Nets import MLP, CNNMnist, CNNEmnist, Res_Model
from models.Fed import FedAvg, FedAvg_zeta
from models.test import test_img
from utils.satcom_helper import *
from utils.__init__ import *


class Sat_Train_Info:
    def __init__(self):
        self.orbit_id = 0
        self.model_round_id = 0


def read_orbits_order(filename, clients):
    try:
        #打开文件
        fp = open(filename,"r")
        line_no = 1
        idx = 0
        for line in fp.readlines():
            if line_no == 1:    # 第一行是参与训练的轨道列表
                line = line.replace('\n','')
                line = line.split(',')
                #以逗号为分隔符把数据转化为列表
                for o in line:
                    if o[0] == '[':
                        o = o[1:]
                    if o[-1] == ']':
                        o = o[:-1]
                    clients[idx].orbit_id = int(o)
                    idx += 1
                line_no += 1
                idx = 0
            elif line_no == 2:  # 第二行是参与训练的轨道对应轮数
                line = line.replace('\n','')
                line = line.split(',')
                for e in line:
                    if e[0] == '[':
                        e = e[1:]
                    if e[-1] == ']':
                        e = e[:-1]
                    clients[idx].model_round_id = int(e)
                    idx += 1
        fp.close()
    except IOError:
        print("file open error, %s is not existing" % filename)


def find_model_round_id_of_orbit(clients, orbit_id):
    for client in clients:
        if client.orbit_id == orbit_id:
            return client.model_round_id
    return -1
        

def vir_comp_main(args, dataset_train, dataset_test, dict_users_train, dict_users_test, net_glob, fraction, gid):   
    clients = [Sat_Train_Info() for i in range(math.ceil(total_orbits_cnt * fraction))]
    # clients = [Sat_Train_Info() for i in range(total_orbits_cnt)]
    # 从文件读取参与训练的客户端列表 & 模型编号
    read_orbits_order("./output_info/participants.txt", clients)

    if os.path.exists('./save/model_w'+str(gid-1)+'.pth'):
        network_state_dict = torch.load('./save/model_w'+str(gid-1)+'.pth')
        net_glob.load_state_dict(network_state_dict)
    # print(net_glob)
    net_glob.train()

    # 假设服务器有所有测试数据，虚拟测试全局模型的准确率(为了验证局部聚合出的模型是否会影响全局性能)
    net_glob.eval()
    acc_train, _ = test_img(net_glob, dataset_train, args)
    acc_test_glob, _ = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f} based on global model in previous round".format(acc_train))
    print("Testing accuracy: {:.2f} based on global model in previous round".format(acc_test_glob))

    # copy weights
    w_glob = net_glob.state_dict()

    loss_train_locals = []
    acc_test_locals = []
    idxs_users = []
    
    if fraction == 1.0:
        w_locals = [w_glob for i in range(args.num_users)] 
    else:
        w_locals = []
        
    # 生成这一轮参与训练的客户端
    for client in clients:
        # print(len(clients), client.orbit_id)
        idxs_users.extend(range(client.orbit_id * total_planes_cnt, (client.orbit_id + 1) * total_planes_cnt))
    
    for idx in idxs_users:
        # 先计算该全局模型在本地测试数据集上的准确率，暂时不计算，太耗时
        localtest = LocalTest(args=args, dataset=dataset_test, idxs=dict_users_test[idx])
        acc, _ = localtest.test(net=copy.deepcopy(net_glob).to(args.device))
        acc_test_locals.append(copy.deepcopy(acc))

        # 再计算该全局模型再本地训练数据集上的更新模型
        orbit_id = idx // total_planes_cnt
        model_round_id = find_model_round_id_of_orbit(clients, orbit_id) 

        # 开始训练
        localtrain = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
        own_net = copy.deepcopy(net_glob)
        if os.path.exists('./save/model_w'+str(model_round_id)+'.pth'):
            own_network_state_dict = torch.load('./save/model_w'+str(model_round_id)+'.pth')                
            own_net.load_state_dict(own_network_state_dict)

        w, loss = localtrain.train(net=copy.deepcopy(own_net).to(args.device))

        # 训练结束，判断模型新鲜程度
        staleness = gid - model_round_id
        if staleness > 2:  # 模型不新鲜, 直接处理
            # print("遇到了不新鲜的模型, 需要特殊处理")
            w_tmp = FedAvg_zeta(w_glob, w, 0.3)
            w_locals.append(w_tmp)
        else:   # 模型是新鲜的，加入到w_locals 等待平均
            if fraction == 1.0:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
        loss_train_locals.append(copy.deepcopy(loss))

    # update global weights
    w_glob = FedAvg(w_locals)
        
    # copy weight to net_glob
    net_glob.load_state_dict(w_glob)
    torch.save(net_glob.state_dict(), './save/model_w'+str(gid)+'.pth')

    # print accuracy
    acc_test_avg = sum(acc_test_locals) / len(acc_test_locals)
    print('Average accuracy {:.3f} based on global model in previous round'.format(acc_test_avg))
    # print loss
    loss_train_avg = sum(loss_train_locals) / len(loss_train_locals)
    print('Average loss {:.3f} in order to get the model of current round'.format(loss_train_avg))     
     
    return acc_test_avg / 100, acc_test_glob / 100
