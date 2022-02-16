import torch.nn as nn
import torch
import argparse
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import time
import os
import json

class LocationAttention(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.attn = nn.Linear(n_input, 1)

    def forward(self, inputs):
        attn_weight = F.softmax(self.attn(inputs), dim=1)
        attention = torch.bmm(inputs.transpose(1, 2), attn_weight).squeeze(dim=-1)
        return attention

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # d_model // h 仍然是要能整除，换个名字仍然意义不变
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        # Q,K,V计算与变形：
        bsz = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        # Q, K相乘除以scale，这是计算scaled dot product attention的第一步
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # 如果没有mask，就生成一个
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # 然后对Q,K相乘的结果计算softmax加上dropout，这是计算scaled dot product attention的第二步：
        attention = self.do(torch.softmax(energy, dim=-1))

        # 第三步，attention结果与V相乘
        x = torch.matmul(attention, V)

        # 最后将多头排列好，就是multi-head attention的结果了
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)

        return x

class Simulator(nn.Module):

    def __init__(self, item_size, item_num, n_hidden, n_output, dropout_rate, device,
                 n_embedding=16, feedback_size=1, n_previous=5, n_heads=8,
                 use_embedding=False, use_history=False, use_attention=False, attention_type='self'):
        super(Simulator, self).__init__()
        self.use_embedding = True if use_embedding == 1 else False
        self.use_history = True if use_history == 1 else False
        self.use_attention = True if use_attention == 1 else False
        n_MLP_input = item_num * item_size
        self.item_num = item_num
        self.item_size = item_size
        self.feedback_size = feedback_size
        self.attention_type = attention_type

        if self.use_embedding and self.use_history and self.use_attention:
            self.item_embedding = nn.Sequential(nn.Linear(item_size, n_embedding),
                                                nn.Tanh())
            self.feedback_embedding = nn.Sequential(nn.Linear(feedback_size, n_embedding),
                                                    nn.Tanh())
            self.history_gru = torch.nn.GRU(2 * item_num * n_embedding, n_embedding)
            if attention_type == 'self':
                self.item_attn = SelfAttention(n_embedding, n_heads, dropout_rate, device)
                self.history_attn = SelfAttention(n_embedding, n_heads, dropout_rate, device)
                n_MLP_input = (n_previous + item_num) * n_embedding
            else:
                self.item_attn = LocationAttention(n_embedding)
                self.history_attn = LocationAttention(n_embedding)
                n_MLP_input = 2 * n_embedding

        elif self.use_embedding and self.use_history:
            self.item_embedding = nn.Sequential(nn.Linear(item_size, n_embedding),
                                                nn.Tanh())
            self.feedback_embedding = nn.Sequential(nn.Linear(feedback_size, n_embedding),
                                                    nn.Tanh())
            self.history_gru = torch.nn.GRU(2 * item_num * n_embedding, n_embedding)
            n_MLP_input = (n_previous + item_num) * n_embedding

        elif self.use_embedding and self.use_attention:
            self.item_embedding = nn.Sequential(nn.Linear(item_size, n_embedding),
                                                nn.Tanh())
            if attention_type == 'self':
                self.item_attn = SelfAttention(n_embedding, n_heads, dropout_rate, device)
                n_MLP_input = item_num * n_embedding
            else:
                self.item_attn = LocationAttention(n_embedding)
                n_MLP_input = n_embedding

        elif self.use_embedding:
            self.item_embedding = nn.Sequential(nn.Linear(item_size, n_embedding),
                                                nn.Tanh())
            n_MLP_input = item_num * n_embedding
        self.MLP = nn.Sequential(nn.Linear(n_MLP_input, 2 * n_hidden),
                                 nn.Dropout(dropout_rate),
                                 nn.ReLU(),
                                 nn.Linear(2 * n_hidden, n_hidden),
                                 nn.Dropout(dropout_rate),
                                 nn.ReLU(),
                                 nn.Linear(n_hidden, n_output))

    def forward(self, Current, History=None):
        '''
        output the log softmax probability
        Current shape: batch_size, item_num*item_size
        History shape: batch_size, previous_pv_num, item_num, item_size+feedback_size
        '''
        # print(Current.shape)
        if self.use_history:
            assert History is not None
            History_ItemEmbedding = self.item_embedding(History[:, :, :, self.feedback_size:])
            History_FeedbackEmbedding = self.feedback_embedding(History[:, :, :, :self.feedback_size])
            HistoryEmbedding = torch.cat((History_ItemEmbedding, History_FeedbackEmbedding), dim=-1)
            HistoryEmbedding = HistoryEmbedding.view(HistoryEmbedding.shape[0], HistoryEmbedding.shape[1], -1)
            HistoryOutput, _ = self.history_gru(HistoryEmbedding.transpose(0, 1))
            History = HistoryOutput.transpose(0, 1)
            if self.use_attention:
                if self.attention_type == 'self':
                    History = self.history_attn(History, History, History)
                else:
                    History = self.history_attn(History)

        if self.use_embedding:
            Current = self.item_embedding(Current.view(-1, self.item_num, self.item_size))
            if self.use_attention:
                if self.attention_type == 'self':
                    Current = self.item_attn(Current, Current, Current)
                else:
                    Current = self.item_attn(Current)

        Current = Current.contiguous()
        if self.use_history:
            History = History.contiguous()
            out = self.MLP(torch.cat((Current.view(Current.shape[0], -1), History.view(History.shape[0], -1)), dim=-1))
        else:
            out = self.MLP(Current.view(Current.shape[0], -1))

        return out

def load_simulator(input_size=28*6, hidden_size=64, output_size=1):
    simulator = Simulator(item_size=28, item_num=6, n_hidden=32, n_output=2, dropout_rate=0.1, device=1,
                 n_embedding=16, feedback_size=1, n_previous=5, n_heads=8,
                 use_embedding=False, use_history=False, use_attention=False, attention_type='self')
    # simulator = Simulator(input_size, hidden_size, output_size)
    simulator.load_state_dict(torch.load('/Users/jiahaonan/Desktop/ali/open_data/simulator/simulator.pkl'))

    '''
    inputs = np.zeros((6*28))
    inputs = torch.Tensor(inputs)
    outputs = simulator(inputs)
    print(outputs)
    '''

    return simulator

# simulator = load_simulator()
# # [1....2....3....4....5....6....]
# inputs = [[i/6]*28 for i in range(6)]
# inputs = np.reshape(np.array(inputs),(1, 28*6))
# inputs = torch.Tensor(inputs)
# outputs = simulator(inputs).detach().numpy()[0]
# outputs = np.exp(outputs[1]) / (np.exp(outputs[0])+np.exp(outputs[1]))
# print(outputs)

# # [6....5....4....3....2....1....]
# inputs = [[i/6]*28 for i in range(5, -1, -1)]
# inputs = np.reshape(np.array(inputs),(28*6))
# inputs = torch.Tensor(inputs)
# outputs = simulator(inputs)
# print(outputs)