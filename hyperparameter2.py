# -*- coding: utf-8 -*-

import torch
from datetime import datetime
class hyperparameter():
    def __init__(self):
        self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.Learning_rate = 0.0002
        # self.Epoch = 200
        self.Epoch = 2000
        # self.Batch_size = 32
        self.Batch_size = 512
        self.Resume = False
        self.Patience = 50
        self.FC_Dropout = 0.5
        self.test_split = 0.2
        self.validation_split = 0.2
        self.decay_interval = 10
        self.lr_decay = 0.5
        self.weight_decay = 1e-4
        self.embed_dim = 64

        self.protein_kernel = [4, 8, 12]
        self.drug_kernel = [4, 6, 8]
        self.conv = 40
        self.char_dim = 64