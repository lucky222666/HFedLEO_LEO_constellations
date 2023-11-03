#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w_glob, w_base, w_trained, alpha):
    w_avg = copy.deepcopy(w_glob)
    for k in w_avg.keys():
        w_avg[k] = w_glob[k] - alpha * (w_base[k] - w_trained[k])
    return w_avg
