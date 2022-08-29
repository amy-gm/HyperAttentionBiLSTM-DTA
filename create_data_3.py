import random
import os
import json
import pickle
from collections import OrderedDict
# from model2 import AttentionDTA
# from CNN_GRUNET_DTA import CNN_GRUNET_DTA
# # from CNN_LSTM import CNN_LSTM_NET
# # from CNN_LSTM_Transformer import CNN_LSTM_Transformer_NET
#
# # from CNN_Transformer_LSTM import CNN_Transformer_LSTM_NET
# from CNN_LSTM_DTA import CNN_LSTM_DTA_NET
# # from CNN_Transformer import CNN_Transformer_NET
# # from Transformer_CNN_DTA import *
# from dataset2 import CustomDataSet, collate_fn
# from torch.utils.data import DataLoader
# from prefetch_generator import BackgroundGenerator
# from tqdm import tqdm
# from rdkit import Chem
# from hyperparameter import hyperparameter
# from pytorchtools import EarlyStopping
# import timeit
# from tensorboardX import SummaryWriter
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from emetrics import *
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc


all_prots = []
# datasets = ['kiba', 'davis']
datasets = ['davis']
for dataset in datasets:
    print('convert data from DeepDTA for ', dataset)

    # train_fold, valid_fold = train_test_fold(dataset)
    fpath = 'data/' + dataset + '/'
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    train_fold = [ee for e in train_fold for ee in e]
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))

    # 药物
    ligands_train = json.load(open(fpath + "ligands_iso.txt"), object_pairs_hook=OrderedDict)
    ligands_test = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    # 蛋白质
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    # 亲和力值
    affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')

    drugs_train = []
    drugs_test = []
    drugs = []
    prots = []
    affinity1 = []
    affinity2 = []
    for d in ligands_train.keys():
        # lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)

        drugs_train.append(ligands_train[d])

        # drugs.append(lg)
    for h in ligands_test.keys():
        # lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        # drugs_train.append(lg)

        drugs_test.append(ligands_test[h])


    for t in proteins.keys():
        prots.append(proteins[t])

    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    opts = ['train', 'test']
    for opt in opts:
        # affinity = np.asarray(affinity)
        # affinity = [-np.log10(y / 1e9) for y in affinity]

        rows, cols = np.where(np.isnan(affinity) == False)

        if opt == 'train':

            # if dataset == 'davis':

            # affinity = [-np.log10(y / 1e9) for y in affinity]
            # affinity = np.asarray(affinity)
            #
            # rows, cols = np.where(np.isnan(affinity) == False)


            rows, cols = rows[train_fold], cols[train_fold]
            with open('data/' + dataset + '_' + opt + '.txt', 'w') as f:
            # f.write('\n')
                for pair_ind in range(len(rows)):
                    ls = []
                    ls += [drugs_train[rows[pair_ind]]]
                    ls += [prots[cols[pair_ind]]]
                    ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                    f.write('\t'.join(map(str, ls)) + '\n')


        elif opt == 'test':

            # affinity = np.asarray(affinity)
            #
            # rows, cols = np.where(np.isnan(affinity) == False)
            rows, cols = rows[valid_fold], cols[valid_fold]
            with open('data/' + dataset + '_' + opt + '.txt', 'w') as f:
            # f.write('\n')
                for pair_ind in range(len(rows)):
                    ls = []
                    ls += [drugs_test[rows[pair_ind]]]
                    ls += [prots[cols[pair_ind]]]
                    ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                    f.write('\t'.join(map(str, ls)) + '\n')


        # with open('data/' + dataset + '_' + opt + '.txt', 'w') as f:
        #     # f.write('\n')
        #     for pair_ind in range(len(rows)):
        #         ls = []
        #         ls += [drugs[rows[pair_ind]]]
        #         ls += [prots[cols[pair_ind]]]
        #         ls += [affinity[rows[pair_ind], cols[pair_ind]]]
        #         f.write('\t'.join(map(str, ls)) + '\n')
    print('\ndataset:', dataset)
    print('train_fold:', len(train_fold))
    print('test_fold:', len(valid_fold))
    print('len(set(drugs_train)),len(set(drugs_test)),len(set(prots)):', len(set(drugs_train)), len(set(drugs_test)), len(set(prots)))
    all_prots += list(set(prots))


