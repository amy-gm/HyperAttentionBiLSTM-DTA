# -*- coding: utf-8 -*-

import random
from CNN_BiLSTMNET_DTA import CNN_BiLSTMNET_DTA
from dataset2 import CustomDataSet, collate_fn
from torch.utils.data import DataLoader
from hyperparameter import hyperparameter
# from hyperparameter2 import hyperparameter
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
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
            test_data_list_test = f.read().strip().split('\n')
        print("load finished")
    elif DATASET == "KIBA":
        dir_input = ('./data/{}.txt'.format(DATASET))
        print("load data")
        with open(dir_input, "r") as f:
            train_data_list = f.read().strip().split('\n')
        print("load finished")


    # random shuffle
    print("data shuffle")
    train_dataset = shuffle_dataset(train_data_list_train, SEED)
    test_dataset = shuffle_dataset(test_data_list_test, SEED)
    TVdataset = CustomDataSet(train_dataset)
    test_dataset = CustomDataSet(test_dataset)
    train_size = len(TVdataset)
    train_dataset_load = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0,
                                    collate_fn=collate_fn, drop_last=True)
    test_dataset_load = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                   collate_fn=collate_fn, drop_last=True)

    model = CNN_BiLSTMNET_DTA(hp).cuda()
    model_st = CNN_BiLSTMNET_DTA.__name__

    """weight initialize"""
    weight_p, bias_p = [], []
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    """load trained model"""
    Loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.Learning_rate)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate * 10,
                                            cycle_momentum=False,
                                            step_size_up=train_size // hp.Batch_size)

    """Start training."""
    print('Training...')

    best_mse = 1000
    best_test_mse = 1000
    best_test_ci = 0
    best_epoch = -1
    model_file_name = 'model_' + model_st + '_' + DATASET + '_dta.model'
    result_file_name = 'result_' + model_st + '_' + DATASET + '_dta.csv'
    """ create model"""

    for epoch in range(1, hp.Epoch + 1):

        print('Training on {} samples...'.format(len(train_dataset_load.dataset)))
        model.train()
        for train_i, train_data in enumerate(train_dataset_load):
            '''data preparation '''
            trian_compounds, trian_proteins, trian_labels = train_data
            trian_compounds = trian_compounds.cuda()
            trian_proteins = trian_proteins.cuda()
            trian_labels = trian_labels.cuda()

            optimizer.zero_grad()

            predicted_interaction = model(trian_compounds, trian_proteins)
            predicted_interaction = predicted_interaction.to(torch.float32)
            trian_labels = trian_labels.to(torch.float32)

            predicted_interaction = predicted_interaction.squeeze()

            train_loss = Loss(predicted_interaction, trian_labels)
            # train_losses_in_epoch.append(train_loss.item())
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            if train_i % 20 == 0:
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                               train_i * len(trian_proteins),
                                                                               len(train_dataset_load.dataset),
                                                                               100. * train_i / len(train_dataset_load),
                                                                               train_loss.item()))


        G, P = predicting(model, test_dataset_load)
        ret = [get_rmse(G, P), get_mse(G, P), get_pearson(G, P), get_spearman(G, P), get_rm2(G, P), best_epoch,
               get_ci(G, P)]

        val = get_mse(G, P)
        if val < best_mse:
            best_mse = val
            train_ci = get_ci(G, P)
            best_epoch = epoch
            torch.save(model.state_dict(), model_file_name)


            ret = [get_rmse(G, P), get_mse(G, P), get_pearson(G, P), get_spearman(G, P), get_rm2(G, P), best_epoch,
                   get_ci(G, P)]
            with open(result_file_name, 'w') as f:
                f.write(','.join(map(str, ret)))
            best_test_mse = ret[1]
            best_test_ci = ret[-1]
            print('mse improved at epoch ', best_epoch, '; best_test_mse,best_test_ci:', best_test_mse,
                  best_test_ci, '; best_train_mse,train_ci:', best_mse, train_ci)
            print('\n')
            print(ret[0], ',', ret[1], ',', ret[2], ',', ret[3], ',', ret[4], ',', ret[6])
        else:
            print(ret[1], 'No improvement since epoch ', best_epoch, '; best_test_mse,best_test_ci:',
                  best_test_mse, best_test_ci)
            print('\n')
            print(ret[0], ',', ret[1], ',', ret[2], ',', ret[3], ',', ret[4], ',', ret[6])







