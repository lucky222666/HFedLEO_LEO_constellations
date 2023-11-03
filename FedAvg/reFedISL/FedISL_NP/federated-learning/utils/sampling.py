#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import random
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# 优化论文 Non-IID版本
def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    # print(idxs_labels[0])
    # print(idxs_labels[1])
    # print(idxs_labels[2])
    idxs_labels = idxs_labels[:,idxs_labels[1, : ].argsort()]
    idxs = idxs_labels[0,:]
    # print(idxs_labels[0])
    # print(idxs_labels[1])
    # print(idxs[0])

    # divide and assign
    idxs_40 = idxs[ : 24000]
    idxs_60 = idxs[24000: ]
    
    # divide and assign
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs_40, all_idxs_60= {}, [i for i in idxs_40], [i for i in idxs_60]
    # ndarray -> list
    random.shuffle(all_idxs_40)
    random.shuffle(all_idxs_60)
    
    for i in range(num_users):
        if i < num_users * 0.4 :
            dict_users[i] = set(np.random.choice(all_idxs_40, num_items, replace=False))
            all_idxs_40 = list(set(all_idxs_40) - dict_users[i])
        else :
            dict_users[i] = set(np.random.choice(all_idxs_60, num_items, replace=False))
            all_idxs_60 = list(set(all_idxs_60) - dict_users[i]) 
    return dict_users


# def mnist_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:,idxs_labels[1, : ].argsort()]
#     idxs = idxs_labels[0,:]

#     # divide and assign
#     idxs_50_1 = idxs[ : 30000]
#     idxs_50_2 = idxs[30000: ]
    
#     # divide and assign
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs_50_1, all_idxs_50_2= {}, [i for i in idxs_50_1], [i for i in idxs_50_2]
#     # ndarray -> list
#     random.shuffle(all_idxs_50_1)
#     random.shuffle(all_idxs_50_2)
    
#     for i in range(num_users):
#         if i < num_users * 0.5 :
#             dict_users[i] = set(np.random.choice(all_idxs_50_1, num_items, replace=False))
#             all_idxs_50_1 = list(set(all_idxs_50_1) - dict_users[i])
#         else :
#             dict_users[i] = set(np.random.choice(all_idxs_50_2, num_items, replace=False))
#             all_idxs_50_2 = list(set(all_idxs_50_2) - dict_users[i]) 
#     return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# 优化论文 Non-IID版本（参考文献6未考虑CIFAR-10数据集的Non-IID版本）
def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = [data[1] for data in dataset]

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # print(idxs_labels[0])
    # print(idxs_labels[1])
    
    idxs_40 = idxs[ : 20000]
    idxs_60 = idxs[20000: ]
    # print(len(idxs_40))
    # print(len(idxs_60))
    # print(type(idxs_40))
    # for i in idxs_40:
    #     print(i)
    
    # divide and assign
    num_items = int(len(dataset)/num_users)
    # dict_users, all_idxs_40, all_idxs_60= {}, [i for i in range(int(len(dataset)*0.4))], [i for i in range(int(len(dataset)*0.6))]
    dict_users, all_idxs_40, all_idxs_60= {}, [i for i in idxs_40], [i for i in idxs_60]
    # print(type(all_idxs_40))
    # print(all_idxs_40)
    # ndarray -> list
    random.shuffle(all_idxs_40)
    random.shuffle(all_idxs_60)
    
    
    for i in range(num_users):
        if i < num_users * 0.4 :
            dict_users[i] = set(np.random.choice(all_idxs_40, num_items, replace=False))
            all_idxs_40 = list(set(all_idxs_40) - dict_users[i])
        else :
            dict_users[i] = set(np.random.choice(all_idxs_60, num_items, replace=False))
            all_idxs_60 = list(set(all_idxs_60) - dict_users[i]) 
    return dict_users


if __name__ == '__main__':
    dataset_train1 = datasets.MNIST('../../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    dataset_train2 = datasets.CIFAR10('../../data/cifar/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 10
    d1 = mnist_noniid(dataset_train1, num)
    d2 = cifar_noniid(dataset_train2, num)
