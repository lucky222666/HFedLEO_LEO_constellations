#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
import random
from torchvision import datasets, transforms


def iid(dataset_train, dataset_test, num_users):
    num_items_train = int(len(dataset_train) / num_users)
    num_items_test = int(len(dataset_test) / num_users)

    dict_users_train, all_idxs_train = {}, [i for i in range(len(dataset_train))]
    dict_users_test, all_idxs_test = {}, [i for i in range(len(dataset_test))]

    for i in range(num_users):
        dict_users_train[i] = set(np.random.choice(all_idxs_train, num_items_train, replace=False))
        dict_users_test[i] = set(np.random.choice(all_idxs_test, num_items_test, replace=False))
        all_idxs_train = list(set(all_idxs_train) - dict_users_train[i])
        all_idxs_test = list(set(all_idxs_test) - dict_users_test[i])
    return dict_users_train, dict_users_test


def noniid(dataset_train, dataset_test, num_users, percent):
    dict_users_train = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}

    num_shards_train, num_imgs_train = num_users * 3, 147     #不够就缺失
    idx_shard_train = [i for i in range(num_shards_train)]
    idxs_train = np.arange(len(dataset_train))
    labels_train = dataset_train.train_labels.numpy()

    num_shards_test, num_imgs_test = num_users * 1, 73
    idx_shard_test = [i for i in range(num_shards_test)]
    idxs_test = np.arange(len(dataset_test))
    labels_test = dataset_test.test_labels.numpy()

    # sort labels
    idxs_labels_train = np.vstack((idxs_train, labels_train))
    idxs_labels_train = idxs_labels_train[:,idxs_labels_train[1, : ].argsort()]
    idxs_train = idxs_labels_train[0,:]
    dict_users_train = {}

    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:,idxs_labels_test[1, : ].argsort()]
    idxs_test = idxs_labels_test[0,:]
    dict_users_test = {}
    
    d = 0   # test集合直接获取 按照73截断
    for i in range(num_users):
        # 如果有客户端抽到了4748-4752这五个数中的任何一个 可能就会数据不足441个
        a, b, c = set(np.random.choice(idx_shard_train, 3, replace=False))
        # d = set(np.random.choice(idx_shard_test, 1, replace=False))
        # print("这是我选的d", d, type(d))
        idx_shard_train.remove(a)
        idx_shard_train.remove(b)
        idx_shard_train.remove(c)
        # idx_shard_test.remove(d)
        # print(a, b, c)
        a = a * num_imgs_train
        b = b * num_imgs_train
        c = c * num_imgs_train
        # d = d * num_imgs_test
        dict_users_train[i] = [i for i in range(3 * num_imgs_train)]
        dict_users_test[i] = [i for i in range(1 * num_imgs_test)]

        point = 0
        j = 0
        flag = 0
        for j in range(num_imgs_train):
            if a >= len(dataset_train):
                point = j
                flag = 1
                break
            dict_users_train[i][j] = idxs_train[a]
            a += 1
        if flag == 0:
            point = j+1
        flag = 0

        for j in range(point, point + num_imgs_train):
            if b >= len(dataset_train):
                point = j
                flag = 1
                break
            dict_users_train[i][j] = idxs_train[b]
            b += 1
        if flag == 0:
            point = j+1
        flag = 0

        for j in range(point, point + num_imgs_train):
            if c >= len(dataset_train):
                point = j
                flag = 1
                break
            dict_users_train[i][j] = idxs_train[c]
            c += 1
        if flag == 0:
            point = j+1       
        
        dict_users_train[i] = dict_users_train[i][:point]
       
        for j in range(num_imgs_test):
            dict_users_test[i][j] = idxs_test[d]
            d += 1
        # if len(dict_users[i])> 441:
        #     print(len(dict_users[i]))
    # print(d)
    # print(len(dict_users_test[0]))
    return dict_users_train, dict_users_test


if __name__ == '__main__':
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
    d3 = noniid(dataset_train3, dataset_test3, num, 0.5)
