import torch.nn as nn
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import numpy as np
import copy
from replay_buffer import Replay_Buffer

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
        n_MLP_input = item_num*item_size
        self.item_num = item_num
        self.item_size = item_size
        self.feedback_size = feedback_size
        self.attention_type = attention_type

        if self.use_embedding and self.use_history and self.use_attention:
            self.item_embedding = nn.Sequential(nn.Linear(item_size,n_embedding),
                                    nn.Tanh())
            self.feedback_embedding = nn.Sequential(nn.Linear(feedback_size, n_embedding),
                                nn.Tanh())
            self.history_gru = torch.nn.GRU(2*item_num*n_embedding, n_embedding)
            if attention_type == 'self':
                self.item_attn = SelfAttention(n_embedding, n_heads, dropout_rate, device)
                self.history_attn = SelfAttention(n_embedding, n_heads, dropout_rate, device)
                n_MLP_input = (n_previous+item_num)*n_embedding
            else:
                self.item_attn = LocationAttention(n_embedding)
                self.history_attn = LocationAttention(n_embedding)
                n_MLP_input = 2*n_embedding
            
        elif self.use_embedding and self.use_history:
            self.item_embedding = nn.Sequential(nn.Linear(item_size,n_embedding),
                                    nn.Tanh())
            self.feedback_embedding = nn.Sequential(nn.Linear(feedback_size, n_embedding),
                                nn.Tanh())
            self.history_gru = torch.nn.GRU(2*item_num*n_embedding, n_embedding)
            n_MLP_input = (n_previous+item_num)*n_embedding

        elif self.use_embedding and self.use_attention:
            self.item_embedding = nn.Sequential(nn.Linear(item_size,n_embedding),
                                    nn.Tanh())
            if attention_type == 'self':
                self.item_attn = SelfAttention(n_embedding, n_heads, dropout_rate, device)
                n_MLP_input = item_num*n_embedding
            else:
                self.item_attn = LocationAttention(n_embedding)
                n_MLP_input = n_embedding

        elif self.use_embedding:
            self.item_embedding = nn.Sequential(nn.Linear(item_size,n_embedding),
                        nn.Tanh())
            n_MLP_input = item_num*n_embedding
        self.MLP = nn.Sequential(nn.Linear(n_MLP_input,2*n_hidden),
                                    nn.Dropout(dropout_rate),
                                    nn.ReLU(),
                                    nn.Linear(2*n_hidden,n_hidden),
                                    nn.Dropout(dropout_rate),
                                    nn.ReLU(),
                                    nn.Linear(n_hidden,n_output))
    
    def forward(self, Current, History=None):
        '''
        output the log softmax probability
        Current shape: batch_size, item_num*item_size
        History shape: batch_size, previous_pv_num, item_num, item_size+feedback_size
        '''
        print(Current.shape)
        if self.use_history:
            assert History is not None
            History_ItemEmbedding = self.item_embedding(History[:,:,:,self.feedback_size:])
            History_FeedbackEmbedding = self.feedback_embedding(History[:,:,:,:self.feedback_size])
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

class Simulator_v1(nn.Module):

    def __init__(self,n_input,n_hidden,n_output):
        super(Simulator, self).__init__()
        self.hidden = nn.Sequential(nn.Linear(n_input,2*n_hidden),
                                    nn.ReLU(),
                                    nn.Linear(2*n_hidden,n_hidden),
                                    # nn.Dropout(p=0.5),
                                    nn.ReLU(),
                                    nn.Linear(n_hidden,n_output),
                                    nn.Sigmoid())

    def forward(self, input):
        '''
        output the log softmax probability
        '''
        out = self.hidden(input)
        # out = torch.sigmoid(out)
        return out



class Simulator_v2(nn.Module):

    def __init__(self, n_input, input_num, n_embedding, n_hidden, n_output, dropout_rate):
        super(Simulator_v2, self).__init__()

        self.n_input = n_input
        self.input_num = input_num
        self.item_embedding = nn.Sequential(nn.Linear(n_input,n_embedding),
                                    nn.Tanh())
        self.item_forward = nn.Sequential(nn.Linear(n_input*input_num,n_hidden),
                                    nn.ReLU(),
                                    nn.Linear(n_hidden,n_embedding),
                                    nn.ReLU())
        self.attn = nn.Linear(n_embedding, 1)
        self.hidden = nn.Sequential(nn.Linear(2*n_embedding,2*n_hidden),
                                    nn.Dropout(dropout_rate),
                                    nn.ReLU(),
                                    nn.Linear(2*n_hidden,n_hidden),
                                    nn.Dropout(dropout_rate),
                                    nn.ReLU(),
                                    nn.Linear(n_hidden,n_output),
                                    nn.LogSoftmax(dim=1))

    def attention(self, ItemEmbedding):
        
        attn_weight = F.softmax(self.attn(ItemEmbedding), dim=1) #[batch_size, 6, 1]
        attention = torch.bmm(ItemEmbedding.transpose(1, 2), attn_weight).squeeze() #[batch_size, embedding_size, item_num]*[batch_size, item_num, 1] = [batch_size, embedding_size, 1]

        return attention
        
    def forward(self, inputs):
        '''
        output the log softmax probability
        '''
        MLP = self.item_forward(inputs)
        ItemEmbedding = self.item_embedding(inputs.view(-1, self.input_num, self.n_input))#[batch_size*item_num*item_size]-->[batch_size, item_num, embedding_size]
        hidden = self.attention(ItemEmbedding)
        out = self.hidden(torch.cat((hidden, MLP), dim=-1))
        return out


class Simulator_v3(nn.Module):

    def __init__(self, item_size, feedback_size, item_num, n_embedding, n_hidden, n_output, dropout_rate, history_type='use_pv'):
        super(Simulator_v3, self).__init__()

        self.item_size = item_size
        self.item_num = item_num
        self.feedback_size = feedback_size
        self.history_type = history_type
        self.item_embedding = nn.Sequential(nn.Linear(item_size,n_embedding),
                                    nn.Tanh())
        self.feedback_embedding = nn.Sequential(nn.Linear(feedback_size, n_embedding),
                            nn.Tanh())
        if self.history_type == 'use_pv':
            self.history_gru = torch.nn.GRU(2*self.item_num*n_embedding, n_embedding)
        elif self.history_type == 'use_item':
            self.history_gru = torch.nn.GRU(2*n_embedding, n_embedding)
        self.item_attn = nn.Linear(n_embedding, 1)
        self.history_attn = nn.Linear(n_embedding, 1)
        self.hidden_with_history = nn.Sequential(nn.Linear(2*n_embedding,2*n_hidden),
                                    nn.Dropout(dropout_rate),
                                    nn.ReLU(),
                                    nn.Linear(2*n_hidden,n_hidden),
                                    nn.Dropout(dropout_rate),
                                    nn.ReLU(),
                                    nn.Linear(n_hidden,n_output))
        self.hidden_without_history = nn.Sequential(nn.Linear(self.item_num*n_embedding,2*n_hidden),
                            nn.Dropout(dropout_rate),
                            nn.ReLU(),
                            nn.Linear(2*n_hidden,n_hidden),
                            nn.Dropout(dropout_rate),
                            nn.ReLU(),
                            nn.Linear(n_hidden,n_output))

    def item_attention(self, ItemEmbedding):
        
        attn_weight = F.softmax(self.item_attn(ItemEmbedding), dim=1) #[batch_size, 6, 1]
        item_attention = torch.bmm(ItemEmbedding.transpose(1, 2), attn_weight).squeeze(dim=-1) #[batch_size, embedding_size, item_num]*[batch_size, item_num, 1] = [batch_size, embedding_size, 1]

        return item_attention

    def history_attention(self, History):
        
        attn_weight = F.softmax(self.history_attn(History), dim=1) #[batch_size, 6, 1]
        history_attention = torch.bmm(History.transpose(1, 2), attn_weight).squeeze(dim=-1) #[batch_size, embedding_size, item_num]*[batch_size, item_num, 1] = [batch_size, embedding_size, 1]

        return history_attention
        
    def forward(self, current, history):
        '''
        output the log softmax probability
        '''
        # print(current.shape)
        Current_ItemEmbedding = self.item_embedding(current.view(-1, self.item_num, self.item_size))#[batch_size*item_num*item_size]-->[batch_size, item_num, embedding_size]
        CurrentAttn = Current_ItemEmbedding.view(Current_ItemEmbedding.shape[0], -1)
        # CurrentAttn = self.item_attention(Current_ItemEmbedding)

        if history is not None:
            if self.history_type == 'use_pv':
                History_ItemEmbedding = self.item_embedding(history[:,:,:,self.feedback_size:])
                History_FeedbackEmbedding = self.feedback_embedding(history[:,:,:,:self.feedback_size])
            elif self.history_type == 'use_item':
                History_ItemEmbedding = self.item_embedding(history[:,:,self.feedback_size:])
                History_FeedbackEmbedding = self.feedback_embedding(history[:,:,:self.feedback_size])
            HistoryEmbedding = torch.cat((History_ItemEmbedding, History_FeedbackEmbedding), dim=-1)
            HistoryEmbedding = HistoryEmbedding.view(HistoryEmbedding.shape[0], HistoryEmbedding.shape[1], -1)
            HistoryOutput, _ = self.history_gru(HistoryEmbedding.transpose(0, 1))
            HistoryOutput = HistoryOutput.transpose(0, 1)
            HistoryAttn = self.history_attention(HistoryOutput)

            out = self.hidden_with_history(torch.cat((CurrentAttn, HistoryAttn), dim=-1))
        else:
            out = self.hidden_without_history(CurrentAttn)
        return out

class Simulator_v4(nn.Module):

    def __init__(self, item_size, feedback_size, item_num, n_embedding, n_hidden, n_output, dropout_rate):
        super(Simulator_v4, self).__init__()

        self.item_size = item_size
        self.item_num = item_num
        self.feedback_size = feedback_size
        self.item_embedding = nn.Sequential(nn.Linear(item_size,n_embedding),
                                    nn.Tanh())
        self.feedback_embedding = nn.Sequential(nn.Linear(feedback_size, n_embedding),
                            nn.Tanh())
        self.history_gru = torch.nn.GRU(2*self.item_num*n_embedding, n_embedding)
        self.item_forward = nn.Sequential(nn.Linear(item_num*n_embedding, n_hidden),
                            nn.ReLU(),
                            nn.Linear(n_hidden,n_embedding),
                            nn.ReLU())
        self.history_attn = nn.Linear(n_embedding, 1)
        self.hidden = nn.Sequential(nn.Linear(2*n_embedding,2*n_hidden),
                                    nn.Dropout(dropout_rate),
                                    nn.ReLU(),
                                    nn.Linear(2*n_hidden,n_hidden),
                                    nn.Dropout(dropout_rate),
                                    nn.ReLU(),
                                    nn.Linear(n_hidden,n_output),
                                    nn.LogSoftmax(dim=1))


    def history_attention(self, History):
        
        attn_weight = F.softmax(self.history_attn(History), dim=1) #[batch_size, 6, 1]
        history_attention = torch.bmm(History.transpose(1, 2), attn_weight).squeeze(dim=-1) #[batch_size, embedding_size, item_num]*[batch_size, item_num, 1] = [batch_size, embedding_size, 1]

        return history_attention
        
    def forward(self, history, current):
        '''
        output the log softmax probability
        '''
        History_ItemEmbedding = self.item_embedding(history[:,:,:,self.feedback_size:])
        History_FeedbackEmbedding = self.feedback_embedding(history[:,:,:,:self.feedback_size])
        HistoryEmbedding = torch.cat((History_ItemEmbedding, History_FeedbackEmbedding), dim=-1)
        HistoryEmbedding = HistoryEmbedding.view(HistoryEmbedding.shape[0], HistoryEmbedding.shape[1], -1)
        HistoryOutput, _ = self.history_gru(HistoryEmbedding.transpose(0, 1))
        HistoryOutput = HistoryOutput.transpose(0, 1)
        HistoryAttn = self.history_attention(HistoryOutput)

        Current_ItemEmbedding = self.item_embedding(current.view(-1, self.item_num, self.item_size))#[batch_size*item_num*item_size]-->[batch_size, item_num, embedding_size]
        # CurrentAttn = self.item_attention(Current_ItemEmbedding)
        CurrentMLP = self.item_forward(Current_ItemEmbedding.view(Current_ItemEmbedding.shape[0], -1))

        # out = self.hidden(torch.cat((CurrentAttn, HistoryAttn), dim=-1))
        out = self.hidden(torch.cat((CurrentMLP, HistoryAttn), dim=-1))
        return out

class Simulator_v5(nn.Module):

    def __init__(self, item_size, feedback_size, item_num, n_embedding, n_hidden, n_output, n_previous_pv, n_heads, dropout_rate, device):
        super(Simulator_v5, self).__init__()

        self.item_size = item_size
        self.item_num = item_num
        self.feedback_size = feedback_size
        self.item_embedding = nn.Sequential(nn.Linear(item_size,n_embedding),
                                    nn.Tanh())
        self.feedback_embedding = nn.Sequential(nn.Linear(feedback_size, n_embedding),
                            nn.Tanh())
        self.history_gru = torch.nn.GRU(2*self.item_num*n_embedding, n_embedding)
        self.item_attn = SelfAttention(n_embedding, n_heads, dropout_rate, device)
        self.history_attn = SelfAttention(n_embedding, n_heads, dropout_rate, device)
        self.hidden = nn.Sequential(nn.Linear((n_previous_pv+item_num)*n_embedding,2*n_hidden),
                                    nn.Dropout(dropout_rate),
                                    nn.ReLU(),
                                    nn.Linear(2*n_hidden,n_hidden),
                                    nn.Dropout(dropout_rate),
                                    nn.ReLU(),
                                    nn.Linear(n_hidden,n_output),
                                    nn.LogSoftmax(dim=1))
        
    def forward(self, current, history):
        '''
        output the log softmax probability
        current shape: batch_size, item_num*item_size
        history shape: batch_size, previous_pv_num, item_num, item_size+feedback_size
        '''
        History_ItemEmbedding = self.item_embedding(history[:,:,:,self.feedback_size:])
        History_FeedbackEmbedding = self.feedback_embedding(history[:,:,:,:self.feedback_size])
        HistoryEmbedding = torch.cat((History_ItemEmbedding, History_FeedbackEmbedding), dim=-1)
        HistoryEmbedding = HistoryEmbedding.view(HistoryEmbedding.shape[0], HistoryEmbedding.shape[1], -1)
        HistoryOutput, _ = self.history_gru(HistoryEmbedding.transpose(0, 1))
        HistoryOutput = HistoryOutput.transpose(0, 1)
        HistoryAttn = self.history_attn(HistoryOutput, HistoryOutput, HistoryOutput)
        # print(HistoryAttn.shape)

        Current_ItemEmbedding = self.item_embedding(current.view(-1, self.item_num, self.item_size))#[batch_size*item_num*item_size]-->[batch_size, item_num, embedding_size]
        CurrentAttn = self.item_attn(Current_ItemEmbedding, Current_ItemEmbedding, Current_ItemEmbedding)
        # print(CurrentAttn.shape)

        out = self.hidden(torch.cat((CurrentAttn.view(CurrentAttn.shape[0], -1), HistoryAttn.view(HistoryAttn.shape[0], -1)), dim=-1))
        # out = self.hidden(torch.cat((CurrentMLP, HistoryAttn), dim=-1))
        return out

class ANet(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim,device):

        super(ANet,self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.device = device

        self.fc1 = nn.Linear(s_dim,30)
        self.fc1.weight.data.normal_(0,0.1) # initialization
        self.fc2 = nn.Linear(6*a_dim,30)
        self.fc2.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(60,a_dim)
        self.out.weight.data.normal_(0,0.1) # initialization

    def forward(self, s):
        candidate, selected = s[:, :self.a_dim*16], s[:, self.a_dim*16:]
        # candidate, selected = s
        # print(candidate.shape, selected.shape)

        c = F.relu(self.fc1(candidate))
        s = F.relu(self.fc2(selected))
        x = self.out(torch.cat((c, s), dim=-1))
        
        x = F.relu(x)
        return x

class CNet(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim,device):

        super(CNet,self).__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.device = device

        self.fcs1 = nn.Linear(s_dim,30)
        self.fcs1.weight.data.normal_(0,0.1) # initialization
        self.fcs2 = nn.Linear(6*a_dim,30)
        self.fcs2.weight.data.normal_(0,0.1) # initialization
        self.fcs3 = nn.Linear(60,30)
        self.fcs3.weight.data.normal_(0,0.1) # initialization
        self.fca = nn.Linear(a_dim,30)
        self.fca.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(60,1)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self,s,a):
        candidate, selected = s[:, :self.a_dim*16], s[:, self.a_dim*16:]
        # print(candidate.shape, selected.shape)

        c = F.relu(self.fcs1(candidate))
        s = F.relu(self.fcs2(selected))
        x = torch.cat((c, s), dim=-1)
        x = F.relu(self.fcs3(x))

        y = F.relu(self.fca(a))
        net = F.relu(torch.cat((x, y), dim=-1))
        q_value = self.out(net)
        return q_value


class DDPG(object):
    def __init__(self, a_dim, s_dim, exploration_rate, lr_c, lr_a, batch_size, memory_capacity, cos_weight, gamma, tau, device, directory):
        self.a_dim, self.s_dim = a_dim, s_dim
        self.epsilon = exploration_rate
        self.var = 0.1
        self.tau = tau
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.cos_weight = cos_weight

        self.Actor_eval = ANet(s_dim,a_dim,self.device).to(self.device)
        self.Actor_target = ANet(s_dim,a_dim,self.device).to(self.device)
        self.Critic_eval = CNet(s_dim,a_dim,self.device).to(self.device)
        self.Critic_target = CNet(s_dim,a_dim,self.device).to(self.device)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(),lr=lr_c)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(),lr=lr_a)
        self.loss_td = nn.MSELoss()
        self.writer = SummaryWriter(directory)
        self.replay_buffer = Replay_Buffer(memory_capacity)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0

    def choose_action(self, state):
        # candidate, selected = self.state_padding(state)
        # candidate = candidate.to(self.device)
        # selected = selected.to(self.device)
        state = self.state_padding(state).unsqueeze(dim=0).to(self.device)
        # action = self.Actor_eval((candidate, selected)).detach().cpu()
        action = self.Actor_eval(state).squeeze().detach().cpu()

        return action

    # def choose_action(self, state):
    #     # candidate = torch.FloatTensor(state.copy())
    #     target = torch.FloatTensor(copy.deepcopy(state[0]))
    #     # print(target.shape)

    #     if np.random.rand()>self.epsilon:
    #         candidate, selected = self.state_padding(state)
    #         candidate = candidate.to(self.device)
    #         selected = selected.to(self.device)
    #         # print(candidate.shape, selected.shape)
    #         proto_action = self.Actor_eval((candidate, selected)).detach().cpu()
    #         # proto_action = self.Actor_eval((candidate, selected)).detach().cpu().numpy()
    #         # print(proto_action.shape)
    #         # proto_action = np.clip(np.random.normal(proto_action, self.var, size=self.a_dim), 0, np.float('inf'))
    #         # proto_action = torch.FloatTensor(proto_action)
    #         # print(proto_action.repeat(candidate.shape[0], 1).shape)
    #         cos_sim = F.cosine_similarity(target, proto_action.repeat(target.shape[0], 1))
    #         # print(cos_sim.shape)
    #         # for index in action_ids:
    #         #     cos_sim[index] = -float('inf')
    #         index = torch.argmax(cos_sim).item()
    #         # action = candidate[index]
    #     else:
    #         index = np.random.randint(0, len(target))
    #         # action = candidate[index]
    #     # self.epsilon *= .9995
    #     # self.var *= .995 
    #     if self.epsilon>0.1:
    #         self.epsilon *= .9995 
    #     # if self.var>0.001:
    #     #     self.var *= .995

    #     return index

    def learn(self):

        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1-self.tau))')
            eval('self.Actor_target.' + x + '.data.add_(self.tau*self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1-self.tau))')
            eval('self.Critic_target.' + x + '.data.add_(self.tau*self.Critic_eval.' + x + '.data)')

        # soft target replacement
        #self.sess.run(self.soft_replace)  # 用ae、ce更新at，ct
        # print('---------------')
        s, s_, pa, a, r, d = self.replay_buffer.sample(self.batch_size)
        bs = torch.FloatTensor(s).squeeze(dim=1).to(self.device)
        # bs = (torch.FloatTensor(s[0]).to(self.device), torch.FloatTensor(s[1]).to(self.device))
        bpa = torch.FloatTensor(pa).to(self.device)
        ba = torch.FloatTensor(a).to(self.device)
        # bs_ = (torch.FloatTensor(s_[0]).to(self.device), torch.FloatTensor(s_[1]).to(self.device))
        bs_ = torch.FloatTensor(s_).squeeze(dim=1).to(self.device)
        br = torch.FloatTensor(r).to(self.device)
        bd = torch.FloatTensor(1-d).to(self.device)
        # print(bs[0].shape, bs[1].shape)
        # print(bs.shape, ba.shape, bs_.shape, br.shape, bd.shape)

        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs,a)  # loss=-q=-ce（s,ae（s））更新ae   ae（s）=a   ae（s_）=a_
        # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
        loss_a = -torch.mean(q) - self.cos_weight*torch.mean(F.cosine_similarity(ba,bpa))
        self.writer.add_scalar('Loss/actor_loss', loss_a, global_step=self.num_actor_update_iteration)
        #print(q)
        #print(loss_a)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(bs_).detach()  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(bs_,a_).squeeze(dim=1).detach()  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        # print(q_.shape)
        q_target = br+bd*self.gamma*q_.detach()  # q_target = 负的
        #print(q_target)
        # q_v = self.Critic_eval(bs,bpa).squeeze(dim=1)
        q_v = self.Critic_eval(bs,ba).squeeze(dim=1)
        
        td_error = self.loss_td(q_target,q_v)
        self.writer.add_scalar('Loss/critic_loss', td_error, global_step=self.num_critic_update_iteration)
        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
        #print(td_error)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

        self.num_actor_update_iteration += 1
        self.num_critic_update_iteration += 1

        return loss_a.item()
        # print('Actor_loss:{:.4f}'.format(loss_a))
        # print('Critic_loss:{:.4f}'.format(td_error))
    
    def state_padding(self, state):
        candidate, selected = copy.deepcopy(state)
        # print(len(candidate), len(selected))
        # print(candidate.shape, selected.shape)
        candidate = torch.FloatTensor(candidate).reshape(1, -1).squeeze()
        selected = torch.FloatTensor(selected).reshape(1, -1).squeeze()
        candidate = torch.cat((candidate, torch.zeros(self.s_dim-candidate.shape[0])))
        selected = torch.cat((selected, torch.zeros(6*self.a_dim-selected.shape[0])))
        return torch.cat((candidate, selected))
