# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_BiLSTMNET_DTA(nn.Module):
    def __init__(self,hp,
                 protein_MAX_LENGH = 1000,
                 drug_MAX_LENGH = 100,
                 num_filters = 32):
        super(CNN_BiLSTMNET_DTA, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.batch = hp.Batch_size

        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.drug_kernel = hp.drug_kernel
        self.num_filters = num_filters
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.protein_kernel = hp.protein_kernel

        self.protein_embed = nn.Embedding(26, self.dim,padding_idx=0)
        self.drug_embed = nn.Embedding(65, self.dim,padding_idx=0)
        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels= self.conv,  kernel_size=self.drug_kernel[0]),
            nn.BatchNorm1d(self.conv),
            nn.ReLU(),

            nn.Conv1d(in_channels=self.conv, out_channels= self.conv*2,  kernel_size=self.drug_kernel[1]),
            nn.BatchNorm1d(self.conv*2),
            nn.ReLU(),

            nn.Conv1d(in_channels=self.conv*2, out_channels= self.conv*4,  kernel_size=self.drug_kernel[2]),
            nn.BatchNorm1d(self.conv*4),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(self.drug_MAX_LENGH-self.drug_kernel[0]-self.drug_kernel[1]-self.drug_kernel[2]+3)

        self.lstm = nn.LSTM(input_size=64, hidden_size=64, bidirectional=True)
        self.linear = nn.Linear(128, 160)
        # self.ln = nn.LayerNorm(64)

        self.Protein_max_pool = nn.MaxPool1d(1000)

        self.attention_layer = nn.Linear(self.conv*4,self.conv*4)
        # self.attention_layer_bn = nn.BatchNorm1d(self.conv*4)
        self.protein_attention_layer = nn.Linear(self.conv * 4, self.conv * 4)
        # self.protein_attention_layer_bn = nn.BatchNorm1d(self.conv*4)
        self.drug_attention_layer = nn.Linear(self.conv * 4, self.conv * 4)
        # self.drug_attention_layer_bn = nn.BatchNorm1d(self.conv * 4)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv*8, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.out = nn.Linear(512, 1)

    def forward(self, drug, protein):
        drugembed = self.drug_embed(drug)
        drugembed = drugembed.permute(0, 2, 1)


        drugConv = self.Drug_CNNs(drugembed)

        # proteinConv = self.Protein_CNNs(proteinembed)
        # 蛋白质
        proteinembed = self.protein_embed(protein)
        # proteinembed = (2,1000,64)

        # proteinembed = proteinembed.permute(1, 0, 2)
        # proteinembed = seq_len, batch, input_size = (1000,2,64)

        proteinembed = proteinembed.permute(1, 0, 2)
        # 1000,2,64
        # hidden_state = torch.zeros(2, self.batch, 80)
        # hidden_state = hidden_state.cuda()
        # cell_state = torch.zeros(2, self.batch, 80)
        # cell_state = cell_state.cuda()

        hidden_state = torch.zeros(2, self.batch, 64)
        hidden_state = hidden_state.cuda()
        # hidden_state = self.ln(hidden_state)

        cell_state = torch.zeros(2, self.batch, 64)
        cell_state = cell_state.cuda()
        # cell_state = self.ln(cell_state)


        # proteinConv, (hn, cn) = self.lstm(proteinembed, (h0, c0))
        proteinConv, (final_hidden_state, final_cell_state) = self.lstm(proteinembed, (hidden_state, cell_state))
        # 1000,2,160

        proteinConv = self.linear(self.relu(proteinConv))


        proteinConv = proteinConv.permute(1, 2, 0)
        # proteinConv = (2, 160,1000)

        drug_att = self.drug_attention_layer(drugConv.permute(0, 2, 1))
        # drug_att = self.drug_attention_layer_bn(drug_att)
        protein_att = self.protein_attention_layer(proteinConv.permute(0, 2, 1))
        # protein_att = self.protein_attention_layer_bn(protein_att)

        d_att_layers = torch.unsqueeze(drug_att, 2).repeat(1, 1, proteinConv.shape[-1], 1)  # repeat along protein size
        p_att_layers = torch.unsqueeze(protein_att, 1).repeat(1, drugConv.shape[-1], 1, 1)  # repeat along drug size
        Atten_matrix = self.attention_layer(self.relu(d_att_layers + p_att_layers))
        # Atten_matrix = self.attention_layer_bn(Atten_matrix)

        Compound_atte = torch.mean(Atten_matrix, 2)
        Protein_atte = torch.mean(Atten_matrix, 1)
        Compound_atte = self.sigmoid(Compound_atte.permute(0, 2, 1))
        Protein_atte = self.sigmoid(Protein_atte.permute(0, 2, 1))

        drugConv = drugConv * 0.5 + drugConv * Compound_atte
        proteinConv = proteinConv * 0.5 + proteinConv * Protein_atte

        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        # drugConv = self.Drug_max_pool(drugConv)
        # drugConv = drugConv.squeeze(2)
        proteinConv = self.relu(proteinConv)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)


        pair = torch.cat([drugConv, proteinConv], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.bn1(fully1)
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.bn2(fully2)
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        fully3 = self.bn3(fully3)
        predict = self.out(fully3)
        return predict

