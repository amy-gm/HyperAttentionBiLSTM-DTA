# -*- coding: utf-8 -*-
"""
@Time:Created on 2020/7/05
@author: Qichang Zhao
"""
import random
import os
import json
import pickle
from collections import OrderedDict

from CNN_BiLSTMNET_DTA import CNN_BiLSTMNET_DTA

from dataset2 import CustomDataSet, collate_fn
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
from hyperparameter import hyperparameter

import numpy as np
import torch
from emetrics import *

def predicting(model, loader):
    model.eval()
    G = torch.Tensor()
    P = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            valid_compounds, valid_proteins, valid_labels = data
            valid_compounds = valid_compounds.cuda()
            valid_proteins = valid_proteins.cuda()
            valid_labels = valid_labels.cuda()
            valid_scores = model(valid_compounds, valid_proteins)
            valid_scores = valid_scores.to(torch.float32)
            valid_labels = valid_labels.to(torch.float32)
            valid_scores = valid_scores.squeeze()
            G = torch.cat((G, valid_labels.cpu()), 0)
            P = torch.cat((P, valid_scores.cpu()), 0)
    return G.numpy().flatten(), P.numpy().flatten()

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

import os
torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == "__main__":
    """select seed"""
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True

    """init hyperparameters"""
    hp = hyperparameter()

    """Load preprocessed data."""
    DATASET = "davis"
    # DATASET = "KIBA"
    print("Train in " + DATASET)

    if DATASET == "davis":
        dir_input_train = ('./data/{}_train.txt'.format(DATASET))
        dir_input_test = ('./data/{}_test.txt'.format(DATASET))
        print("load data")
        with open(dir_input_train, "r") as f:
            train_data_list_train = f.read().strip().split('\n')
        with open(dir_input_test, "r") as f:
            train_data_list_test = f.read().strip().split('\n')
        print("load finished")
    elif DATASET == "KIBA":
        dir_input = ('./data/{}.txt'.format(DATASET))
        print("load data")
        with open(dir_input, "r") as f:
            train_data_list = f.read().strip().split('\n')
        print("load finished")


    # random shuffle
    print("data shuffle")
    test_dataset = shuffle_dataset(train_data_list_test, SEED)
    test_dataset = CustomDataSet(test_dataset)
    test_dataset_load = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                   collate_fn=collate_fn, drop_last=True)


    model = CNN_BiLSTMNET_DTA(hp).cuda()
    model_st = CNN_BiLSTMNET_DTA.__name__
    result = []
    model_file_name = 'model_' + model_st + '_' + DATASET + '_dta.model'
    if os.path.isfile(model_file_name):
        model.load_state_dict(torch.load(model_file_name), strict=False)
        G, P = predicting(model, test_dataset_load)
        ret = [get_rmse(G, P), get_mse(G, P), get_pearson(G, P), get_spearman(G, P), get_rm2(G, P), get_ci(G, P)]
        ret = [DATASET, model_st] + [round(e, 3) for e in ret]
        result += [ret]
        print('dataset,model,rmse,mse,pearson,spearman,ci')
        print(ret)
    else:
        print('model is not available!')
with open('result.csv', 'w') as f:
    f.write('dataset,model,rmse,mse,pearson,spearman,ci\n')
    for ret in result:
        f.write(','.join(map(str, ret)) + '\n')






