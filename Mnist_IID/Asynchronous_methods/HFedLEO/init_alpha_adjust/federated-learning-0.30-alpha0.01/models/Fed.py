#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def FedAvg_zeta(w_global, w_local, zeta):
    w_res = copy.deepcopy(w_global)
   
    for k in w_local.keys():
        w_res[k] = (1- zeta) * w_global[k] + zeta * w_local[k]

    return w_res