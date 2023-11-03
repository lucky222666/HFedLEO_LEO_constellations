#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import random


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

    # Divide and assign for training set（划分训练集)
    idxs_train = np.arange(len(dataset_train))
    labels_train = dataset_train.train_labels.numpy()   # cifar在此可能不适用，原始labels = [data[1] for data in dataset]
    # Sort labels（按标签对训练集索引和标签进行排序）
    idxs_lables_train = np.vstack((idxs_train, labels_train))
    idxs_lables_train = idxs_lables_train[:, idxs_lables_train[1, :].argsort()]
    idxs_train = idxs_lables_train[0, :]

    split_point_train = int(percent * len(dataset_train))
    idxs_train_first = idxs_train[: split_point_train]  # 将前一部分的数据划分给一个组
    idxs_train_second = idxs_train[split_point_train:]  # 将后一部分的数据划分给另一个组


    # Divide and assign for testing set（划分测试集）
    idxs_test = np.arange(len(dataset_test))  # 创建测试集索引的numpy数组
    labels_test = dataset_test.test_labels.numpy()  # 获取测试集标签，并转换为numpy数组
    # Sort labels for testing set（按标签对测试集索引和标签进行排序）
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]  # 根据标签对索引和标签进行排序
    idxs_test = idxs_labels_test[0, :]  # 获取排序后的测试集索引

    split_point_test = int(percent * len(dataset_test))
    idxs_test_first = idxs_test[: split_point_test]  # 将前一部分的数据划分给一个组
    idxs_test_second = idxs_test[split_point_test:]  # 将后一部分的数据划分给另一个组

    # divide and assign
    num_items_train = int(len(dataset_train)/num_users)
    num_items_test = int(len(dataset_test)/num_users)
    dict_users_train, all_idxs_train_first, all_idxs_train_second= {}, [i for i in idxs_train_first], [i for i in idxs_train_second]
    dict_users_test, all_idxs_test_first, all_idxs_test_second ={}, [i for i in idxs_test_first], [i for i in idxs_test_second]
    # ndarray -> list
    random.shuffle(all_idxs_train_first)
    random.shuffle(all_idxs_train_second)
    random.shuffle(all_idxs_test_first)
    random.shuffle(all_idxs_test_second)

    
    for i in range(num_users):
        if i < num_users * percent :
            dict_users_train[i] = set(np.random.choice(all_idxs_train_first, num_items_train, replace=False))
            dict_users_test[i] = set(np.random.choice(all_idxs_test_first, num_items_test, replace=False))
            all_idxs_train_first = list(set(all_idxs_train_first) - dict_users_train[i])
            all_idxs_test_first = list(set(all_idxs_test_first) - dict_users_test[i])
        else :
            dict_users_train[i] = set(np.random.choice(all_idxs_train_second, num_items_train, replace=False))
            dict_users_test[i] = set(np.random.choice(all_idxs_test_second, num_items_test, replace=False))
            all_idxs_train_second = list(set(all_idxs_train_second) - dict_users_train[i]) 
            all_idxs_test_second = list(set(all_idxs_test_second) - dict_users_test[i])
    return dict_users_train, dict_users_test
