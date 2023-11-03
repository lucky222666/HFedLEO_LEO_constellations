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


# SDA Non-IID版本
def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 400, 150
    idx_shard = [i for i in range(num_shards)]
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1, : ].argsort()]
    idxs = idxs_labels[0,:]
    dict_users = {}
    # print(idxs[10:])
    
    for i in range(num_users):
        a, b = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard.remove(a)
        idx_shard.remove(b)
        # print(a, b)
        a = a * num_imgs
        b = b * num_imgs
        dict_users[i] = [i for i in range(2 * num_imgs)]
        for j in range(num_imgs):
            dict_users[i][j] = idxs[a]
            a += 1
        for j in range(num_imgs, 2 * num_imgs):
            dict_users[i][j] = idxs[b]
            b += 1
    return dict_users


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

    num_shards_train, num_imgs_train = 400, 150
    idx_shard_train = [i for i in range(num_shards_train)]
    idxs_train = np.arange(num_shards_train*num_imgs_train)
    labels_train = dataset_train.train_labels.numpy()

    num_shards_test, num_imgs_test = 400, 25
    idx_shard_test = [i for i in range(num_shards_test)]
    idxs_test = np.arange(num_shards_test*num_imgs_test)
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
    
    for i in range(num_users):
        a, b = set(np.random.choice(idx_shard_train, 2, replace=False))
        c, d = set(np.random.choice(idx_shard_test, 2, replace=False))
        idx_shard_train.remove(a)
        idx_shard_train.remove(b)
        idx_shard_test.remove(c)
        idx_shard_test.remove(d)
        # print(a, b)
        a = a * num_imgs_train
        b = b * num_imgs_train
        c = c * num_imgs_test
        d = d * num_imgs_test
        dict_users_train[i] = [i for i in range(2 * num_imgs_train)]
        dict_users_test[i] = [i for i in range(2 * num_imgs_test)]
        for j in range(num_imgs_train):
            dict_users_train[i][j] = idxs_train[a]
            a += 1
        for j in range(num_imgs_train, 2 * num_imgs_train):
            dict_users_train[i][j] = idxs_train[b]
            b += 1

        for j in range(num_imgs_test):
            dict_users_test[i][j] = idxs_test[c]
            c += 1
        for j in range(num_imgs_test, 2 * num_imgs_test):
            dict_users_test[i][j] = idxs_test[d]
            d += 1
    return dict_users_train, dict_users_test