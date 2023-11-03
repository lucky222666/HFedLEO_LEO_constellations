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


# 参考文献6 Non-IID版本
# def mnist_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     # 60,000 training imgs -->  200 imgs/shard X 300 shards
#     num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]

#     # divide and assign 2 shards/client
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate(
#                 (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users


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


def emnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from EMNIST dataset
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


def emnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from EMNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = num_users * 3, 147     #不够就缺失
    idx_shard = [i for i in range(num_shards)]
    # idxs = np.arange(num_shards*num_imgs)
    idxs = np.arange(len(dataset))
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1, : ].argsort()]
    idxs = idxs_labels[0,:]
    dict_users = {}
    
    for i in range(num_users):
        # 如果有客户端抽到了4748-4752这五个数中的任何一个 可能就会数据不足441个
        a, b, c = set(np.random.choice(idx_shard, 3, replace=False))
        idx_shard.remove(a)
        idx_shard.remove(b)
        idx_shard.remove(c)
        # print(a, b, c)
        a = a * num_imgs
        b = b * num_imgs
        c = c * num_imgs
        dict_users[i] = [i for i in range(3 * num_imgs)]
        point = 0
        j = 0
        flag = 0
        for j in range(num_imgs):
            if a >= len(dataset):
                point = j
                flag = 1
                break
            dict_users[i][j] = idxs[a]
            a += 1
        if flag == 0:
            point = j+1
        flag = 0

        for j in range(point, point + num_imgs):
            if b >= len(dataset):
                point = j
                flag = 1
                break
            dict_users[i][j] = idxs[b]
            b += 1
        if flag == 0:
            point = j+1
        flag = 0

        for j in range(point, point + num_imgs):
            if c >= len(dataset):
                point = j
                flag = 1
                break
            dict_users[i][j] = idxs[c]
            c += 1
        if flag == 0:
            point = j+1
        flag = 0
        
        dict_users[i] = dict_users[i][:point]
        # if len(dict_users[i])> 441:
        #     print(len(dict_users[i]))
    return dict_users


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
    # dataset_train1 = datasets.MNIST('../../data/mnist/', train=True, download=True,
    #                                transform=transforms.Compose([
    #                                    transforms.ToTensor(),
    #                                    transforms.Normalize((0.1307,), (0.3081,))
    #                                ]))
    # dataset_train2 = datasets.CIFAR10('../../data/cifar/', train=True, download=True,
    #                                transform=transforms.Compose([
    #                                    transforms.ToTensor(),
    #                                    transforms.Normalize((0.1307,), (0.3081,))
                                #    ]))  
    dataset_train3 = datasets.EMNIST('../../data/emnist/', split="bymerge", train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    
    dataset_test3 = datasets.EMNIST('../../data/emnist/', split="bymerge", train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    
    num = 1584
    # d1 = mnist_noniid(dataset_train1, num)
    # d2 = cifar_noniid(dataset_train2, num)
    # d3 = emnist_noniid(dataset_train3, num)
    print(len(dataset_train3))
    print(len(dataset_test3))
