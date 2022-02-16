from open_data.config import *
import numpy as np
import open_data.data_process as dp
import math
import random
import torch
import copy
from open_data.simulator.load_simulator import load_simulator

file_path = './data/criteo_pv_rec_normalFill.txt'
# 16->6 场景建模
class Env(object):
    def __init__(self):
        self.state = ...
        self.reward = 0
        self.action = 0
        self.label = []

        self.data_cnt = 0
        self.dataset = dp.load_hierachical_data(filename=file_path)
        # self.dataset = dp.load_criteo_far_data()
        # self.dataset = dp.load_data()

        self.state_one_line = []
        self.label_one_line = []
        self.one_line_cnt = 0
        self.k = 6
        self.canditate_size = 16
        self.action_dim = 28

    def to_state(self, data_piece, algo='DQN'):
        X_t = []
        for item in data_piece:
            X_t.extend(item['feature'])

        state = []
        label = []
        for item in data_piece:
            q = item['feature']
            q.extend(X_t)
            state.append(q)

            label.append([eval(item['rel']),eval(item['click']),eval(item['purchase'])])

        # 返回一条包含的所有state（包含16条数据）16 * 476(28*16+16), 其对应的标签label 16 * 3
        # 返回一条包含的所有X_t（包含16条数据）16 * 448(28*16), 其对应的标签label 16 * 3
        self.state_one_line = X_t
        self.label_one_line = label

        X_t = np.array(X_t)
        # np.reshape(state, (-1, state_dim))
        if algo == 'DQN':
            X_t, label = self.shuffle_item(X_t, label)

        return X_t, label

    def shuffle_item(self, X_t, label):
        '''
        shuffle canditate item for DQN, dueling DQN index method
        :param X_t: item*16
        :param label: label*16
        :return:
        '''
        index_list = random.sample(range(0,16), 16)
        shuffle_Xt = np.array([])
        shuffle_label = []
        for i in index_list:
            xt = X_t[i*item_dim: (i+1)*item_dim]
            lt = label[i]

            shuffle_Xt = np.append(shuffle_Xt, xt)
            shuffle_label.append(lt)

        return shuffle_Xt, shuffle_label

    def to_reward(self, price, label=[0,0,0]):
        exp, click, purchase = label
        # print("label: ", label)
        reward = exp + click*5 + purchase*10
        # reward = exp + click + purchase * eval(price) # 奖励考虑成交价格
        return reward

    def reset(self, algo='DDPG'):
        self.one_line_cnt = 0

        head = self.dataset[self.data_cnt]
        self.data_cnt += 1

        self.state, self.label = self.to_state(head, algo=algo)

        state = np.array(self.state)
        np.reshape(state, (-1, state_dim))
        return self.state, self.label

    def step_in_one_line(self, action):
        # print(action)
        self.one_line_cnt += 1
        state_, label = self.state_one_line, self.label_one_line[action]
        price = state_[self.action_dim * action: self.action_dim * action + self.action_dim][-5] # 选中动作的价格
        state_[self.action_dim * action: self.action_dim * action + self.action_dim] = [-1] * self.action_dim
        self.state_one_line = state_

        reward = self.to_reward(price, label)
        done = self.one_line_cnt == self.k

        state_ = np.array(state_)
        np.reshape(state_, (-1, state_dim))
        return state_, reward, done

    def cosine_similarity(self, x, y):
        num = x.dot(y.T)
        denom = np.linalg.norm(x) * np.linalg.norm(y) + 1e-6
        return num / denom

    def match_action_cos(self, action_vec, state):
        Action_prob = np.zeros((self.canditate_size,))
        for i in range(self.canditate_size):
            action_i = state[self.action_dim*i: self.action_dim*(i+1)].astype(np.float)
            if -1 in action_i:
                continue
            cos = self.cosine_similarity(action_vec, action_i)
            Action_prob[i] = cos
        return np.argmax(Action_prob)

    def L2_distance(self, x, y):
        dis = np.linalg.norm(x - y)
        return dis

    def match_action_L2(self, action_vec, state):
        L2_distance_vec = np.full((self.canditate_size,), 1e5)
        for i in range(self.canditate_size):
            action_i = state[self.action_dim * i: self.action_dim * (i + 1)].astype(np.float)
            if -1 in action_i:
                continue
            dis_i = self.L2_distance(action_vec, action_i)
            # print(dis_i)# 测试一下
            L2_distance_vec[i] = dis_i
        return np.argmin(L2_distance_vec)

    def match_action_cluster(self, action_vec, state, cluster, name='kmeans'):
        Action_clus = []
        if name == 'dbscan':
            action = cluster.fit_predict(np.reshape(action_vec, (1, -1)))
        if name == 'kmeans' or name == 'minikmeans':
            action = cluster.predict(np.reshape(action_vec, (1, -1)))

        # find action_vec belong to which cluster
        for i in range(self.canditate_size):
            action_i = state[self.action_dim * i: self.action_dim * (i + 1)].astype(np.float)
            if -1 in action_i or np.sum(action_i)==0:
                continue
            if name == 'dbscan':
                clu_i = cluster.fit_predict(np.reshape(action_i,(1,-1)))
            if name == 'kmeans' or name == 'minikmeans':
                clu_i = cluster.predict(np.reshape(action_i, (1, -1)))
            if clu_i == action:
                Action_clus.append(i)

        # find action belong to which action_index
        max_cos = -1
        max_index = -1
        for i in range(len(Action_clus)):
            index = Action_clus[i]
            action_index = state[self.action_dim * index: self.action_dim * (index + 1)].astype(np.float)
            cos = self.cosine_similarity(action_vec, action_index)
            if cos > max_cos:
                max_cos = cos
                max_index = index

        # TODO: 动作有可能选重复
        if max_index == -1:
            max_index = np.random.randint(0,self.canditate_size)
        return max_index

    def to_ndcg_2D(self, position_vec, action=''):
        '''
        六宫格视的rank序视为purchase=1:=rank=1, click=1:=rank=2...
        :param position_vec: 当前哪些位置防止了哪些item
        :return: 以pv内原始序为ground truth前提下的ndcg
        '''
        pv = self.dataset[self.data_cnt-1]
        # debug_list = []
        # for p in pv:
        #     debug_list.append(p['feature'])
        # debug_list = np.array(debug_list).astype(np.float)
        item_list = []
        for i in range(self.k):
            item = item_vector_match_reward()
            item.original_rank = i
            item.item_feature = np.array(pv[i]['feature']).astype(np.float)
            item.sum_key = np.sum(item.item_feature)
            rel_i = float(pv[i]['rel'])
            item.rel = rel_i

            item_list.append(item)

        # 按照item.rel反向排序后下标即为新序
        item_list = sorted(item_list, key=lambda item: item.rel, reverse=True)

        IDCG = 0
        for i in range(self.k):
            item = item_list[i]
            rel = item.rel
            item.new_rank = i
            IDCG += (2.0**rel-1.0)/ (math.log2(item.new_rank+2.0))

        DCG = 0
        for i in range(self.k):
            action_i = position_vec[i*c_action_dim:(i+1)*c_action_dim]
            # action_i = np.array(pv[i]['feature']).astype(np.float)
            if -1 in action_i: continue
            action_i_sum_key = np.sum(action_i)

            # 在item_list(即goal)中寻找是否能匹配到
            match = False
            match_index = -1
            match_rel = 0
            for j in range(self.k):
                ideal_item = item_list[j]
                if i == ideal_item.original_rank:
                    match_index = ideal_item.new_rank
                if action_i_sum_key == ideal_item.sum_key:
                    match = True
                    match_rel = ideal_item.rel

            if match is False: continue
            # 如果匹配到，则换算过来实际位置i应该对应的重排位置index
            index = match_index
            rel = match_rel
            DCG += (2.0**rel-1) / (math.log2(index+2.0))

        ans = DCG/IDCG
        return ans

# 用于rerank_reward的action类
class item_vector_match_reward(object):
    def __init__(self):
        self.item_feature = []
        self.original_rank = 0
        self.new_rank = 0
        self.rel = 0
        self.sum_key = 0 # 用特征值和代表key

# rerank 场景建模
class Env_rerank(object):
    '''
    state [item_feature*16, goal, position_vec]
    action 0~5
    reward ndcg@6
    state_ [item_feature*16, goal, position_vec[action:action+1]=action_feature]
    '''
    def __init__(self):
        self.state = []
        self.goal = []
        self.reward_last = 0

        self.data_pointer = 0 # 数据集指针

        self.k = 6
        self.canditate_size = 16
        self.action_dim = 28

        self.action_space = []  # 候选动作空间
        self.canditate_set = [] # 待选待放
        self.position_vec = []  # 已选已放

        self.dataset =dp.load_hierachical_data(filename=file_path)
        # self.dataset = dp.load_criteo_far_data()

    def reset(self, state, goal, pointer):
        '''
        :param state: 即observation, item_feature * 16
        :param goal: 上层生成的16->6的结果, selected_item_feature * 6
        :param pointer: 当前处于哪条pv, 对应训练时的episode
        :return: 初始state, 这里使用了state,goal,position_vec的拼接
        '''
        goal = np.reshape(np.array(goal),(-1,))
        self.goal = goal
        self.canditate_set = np.array(goal.copy()).astype('float64')
        # self.state = np.append(state, goal)
        self.state = self.canditate_set
        self.action_space = goal.copy()
        self.position_vec = np.full((c_action_dim * self.k,), -1.0)
        self.data_pointer = pointer
        self.reward_last = 0

        return np.append(self.state, self.position_vec)

    def to_reward(self, position_vec, action='', dynamic=True):
        if dynamic:
            reward = self.to_ndcg_2D(position_vec, action) # 动态NDCG
        else:
            reward = self.to_ndcg_ori_rank(position_vec, action) # 静态NDCG
        return reward

    def to_ndcg_2D(self, position_vec, action=''):
        '''
        六宫格视的rank序视为purchase=1:=rank=1, click=1:=rank=2...
        :param position_vec: 当前哪些位置防止了哪些item
        :return: 以pv内原始序为ground truth前提下的ndcg
        '''
        pv = self.dataset[self.data_pointer]
        # debug_list = []
        # for p in pv:
        #     debug_list.append(p['feature'])
        # debug_list = np.array(debug_list).astype(np.float)
        item_list = []
        for i in range(self.k):
            item = item_vector_match_reward()
            item.original_rank = i
            item.item_feature = np.array(pv[i]['feature']).astype(np.float)
            item.sum_key = np.sum(item.item_feature)
            rel_i = float(pv[i]['rel'])
            item.rel = rel_i

            item_list.append(item)

        # 按照item.rel反向排序后下标即为新序
        item_list = sorted(item_list, key=lambda item: item.rel, reverse=True)

        IDCG = 0
        for i in range(self.k):
            item = item_list[i]
            rel = item.rel
            item.new_rank = i
            IDCG += (2.0**rel-1.0)/ (math.log2(item.new_rank+2.0))

        DCG = 0
        for i in range(self.k):
            action_i = position_vec[i*c_action_dim:(i+1)*c_action_dim]
            # action_i = np.array(pv[i]['feature']).astype(np.float)
            if -1 in action_i: continue
            action_i_sum_key = np.sum(action_i)

            # 在item_list(即goal)中寻找是否能匹配到
            match = False
            match_index = -1
            match_rel = 0
            for j in range(self.k):
                ideal_item = item_list[j]
                if i == ideal_item.original_rank:
                    match_index = ideal_item.new_rank
                if action_i_sum_key == ideal_item.sum_key:
                    match = True
                    match_rel = ideal_item.rel

            if match is False: continue
            # 如果匹配到，则换算过来实际位置i应该对应的重排位置index
            index = match_index
            rel = match_rel
            DCG += (2.0**rel-1) / (math.log2(index+2.0))

        ans = DCG/IDCG
        return ans

    def to_ndcg_ori_rank(self, position_vec, action=''):
        '''
        六宫格视的rank序视为1、2、3、4、5、6
        :param position_vec: 当前哪些位置防止了哪些item
        :return: 以pv内原始序为ground truth前提下的ndcg
        '''
        pv = self.dataset[self.data_pointer]
        # debug_list = []
        # for p in pv:
        #     debug_list.append(p['feature'])
        # debug_list = np.array(debug_list).astype(np.float)
        item_list = []
        for i in range(self.k):
            item = item_vector_match_reward()
            item.item_feature = np.array(pv[i]['feature']).astype(np.float)
            item.sum_key = np.sum(item.item_feature)
            e = pv[i]['exposure']
            c = pv[i]['click']
            p = pv[i]['purchase']
            rel_i = 0
            if p == '1':
                rel_i = 3
            elif c == '1':
                rel_i = 2
            elif e == '1':
                rel_i = 1
            item.rel = rel_i

            item_list.append(item)

        IDCG = 0
        for i in range(self.k):
            item = item_list[i]
            rel = item.rel
            IDCG += (2.0**rel-1.0)/ (math.log2(i+2.0))

        DCG = 0
        for i in range(self.k):
            action_i = position_vec[i*c_action_dim:(i+1)*c_action_dim]
            # action_i = np.array(pv[i]['feature']).astype(np.float)
            if -1 in action_i: continue
            action_i_sum_key = np.sum(action_i)

            # 在item_list(即goal)中寻找是否能匹配到
            match = False
            match_rel = 0
            for j in range(self.k):
                ideal_item = item_list[j]
                if action_i_sum_key == ideal_item.sum_key:
                    match = True
                    match_rel = ideal_item.rel
                    break

            if match is False: continue
            # 如果匹配到，则匹配到对应的rel
            rel = match_rel
            DCG += (2.0**rel-1) / (math.log2(i+2.0))

        ans = DCG/IDCG
        return ans

    def step(self, action, action_feature, pointer):
        '''
        :param action: 0～5(对应2-D的0,0->0,1...->1,2) 把action_feature对应的item放置在position_vec的哪个位置
        :param action_feature: 待放置的item
        :return: 将action_feature放置到position_vec的action位置后能带来的ndcg收益
        '''
        # print(action)
        self.position_vec[action*c_action_dim: (action+1)*c_action_dim] = action_feature
        self.canditate_set[action*c_action_dim: (action+1)*c_action_dim] = [-1] * c_action_dim
        self.data_pointer = pointer
        state_ = np.append(self.state, self.position_vec)

        # reward 使用差分式的ndcg@k+1-ndcg@k
        reward_new = self.to_reward(self.position_vec, action, dynamic=dynamic_reward_flag)
        reward = reward_new - self.reward_last
        self.reward_last = reward_new

        done = True
        if -1 in self.position_vec:
            done = False

        if done:
            baseline_pos = np.array(self.goal).astype(np.float)
            reward_baseline = self.to_reward(baseline_pos)
            done = reward_new, reward_baseline # 最后一个动作返回整体ndcg

        return state_, reward, done

    def cosine_similarity(self, x, y):
        num = x.dot(y.T)
        denom = np.linalg.norm(x) * np.linalg.norm(y) + 1e-6
        return num / denom

    def match_action_cos(self, action_vec):
        state = self.canditate_set
        Action_prob = np.zeros((self.k,))
        for i in range(self.k):
            action_i = state[self.action_dim * i: self.action_dim * (i + 1)].astype(np.float)
            if -1 in action_i:
                continue
            cos = self.cosine_similarity(action_vec, action_i)
            Action_prob[i] = cos
        return np.argmax(Action_prob)

    def match_action_cluster(self, action_vec, cluster, name='kmeans'):
        Action_clus = []
        if name == 'dbscan':
            action = cluster.fit_predict(np.reshape(action_vec,(1,-1)))
        if name == 'kmeans':
            action = cluster.predict(np.reshape(action_vec, (1, -1)))
        # find action_vec belong to which cluster
        for i in range(self.canditate_size):
            action_i = self.canditate_set[c_action_dim * i: c_action_dim * (i + 1)].astype(np.float)
            if -1 in action_i or np.sum(action_i)==0:
                continue
            if name == 'dbscan':
                clu_i = cluster.fit_predict(np.reshape(action_i,(1,-1)))
            if name == 'kmeans':
                clu_i = cluster.predict(np.reshape(action_i, (1, -1)))
            if clu_i == action:
                Action_clus.append(i)

        # find action belong to which action_index
        max_cos = -1
        max_index = -1
        for i in range(len(Action_clus)):
            index = Action_clus[i]
            action_index = self.canditate_set[c_action_dim * index: c_action_dim * (index + 1)].astype(np.float)
            cos = self.cosine_similarity(action_vec, action_index)
            if cos > max_cos:
                max_cos = cos
                max_index = index

        if max_index == -1:
            max_index = np.random.randint(0,self.k)

        return max_index

    def L2_distance(self, x, y):
        dis = np.linalg.norm(x - y)
        return dis

    def match_action_L2(self, action_vec):
        state = self.canditate_set
        L2_distance_vec = np.full((self.k,), 1e5)
        for i in range(self.k):
            action_i = state[self.action_dim * i: self.action_dim * (i + 1)]
            if -1 in action_i:
                continue
            dis_i = self.L2_distance(action_vec, action_i)
            # print(dis_i)# 测试一下
            L2_distance_vec[i] = dis_i
        return np.argmin(L2_distance_vec)

# hla pv预测场景建模
class Env_high_level(object):
    def __init__(self):
        self.data_pointer = 0
        self.dataset = dp.load_hierachical_data(filename=file_path)
        # self.dataset = dp.load_criteo_far_data()
        # self.dataset = dp.load_data()

        self.state_one_line = []
        self.label_one_line = []

        self.k = 6
        self.canditate_size = 16
        self.random_done_prop = 0.1
        self.item_dim = 28
        self.gamma = 0.9 # reward discount

        self.sum_pv_price = 0
        self.simulator = load_simulator()

    def to_state(self, data_piece):
        X_t = []
        for item in data_piece:
            X_t.extend(item['feature'])

        state = []
        label = []
        for item in data_piece:
            q = item['feature']
            q.extend(X_t)
            state.append(q)

            label.append([eval(item['rel']),eval(item['click']),eval(item['purchase'])])

        # 返回一条包含的所有state（包含16条数据）16 * 476(28*16+16), 其对应的标签label 16 * 3
        # 返回一条包含的所有X_t（包含16条数据）16 * 448(28*16), 其对应的标签label 16 * 3
        self.state_one_line = X_t
        self.label_one_line = label

        X_t = np.array(X_t).astype('float64')
        # np.reshape(state, (-1, state_dim))
        return X_t, label

    def to_reward(self, state, label, goal, lla_action, lla_reward_list, mode='train'):
        '''

        :param state: HLA的16个候选item 28*16
        :param label: 16个候选item对应的标签 3*16
        :param goal: HLA给出的proto-pv 28*6
        :param lla_action: LLA组织好的PV 28*6
        :param lla_reward_list: LLA返回的reward列表 1*6
        :return: simulator打分（+）pv与ground truth相似度（+）simulator(action_pv)/simulator(label_pv)
        '''

        # pv sum price
        # for i in range(self.k):
        #     self.sum_pv_price += lla_action[i * self.item_dim: (i + 1) * self.item_dim][-5]

        # simulator对lla-action的pv-wise成交概率打分
        lla_action = np.reshape(lla_action, (1, 28*6))
        lla_simulator_score = self.simulator(torch.Tensor(lla_action)).detach().numpy()[0]
        lla_simulator_score = np.exp(lla_simulator_score[1]) / (np.exp(lla_simulator_score[0]) + np.exp(lla_simulator_score[1]))

        # simulator对hla预测的proto-wise的pv-wise成交概率打分
        goal = np.reshape(goal, (1, 28*6))
        hla_simulator_score = self.simulator(torch.Tensor(goal)).detach().numpy()[0]
        hla_simulator_score = np.exp(hla_simulator_score[1]) / (np.exp(hla_simulator_score[0]) + np.exp(hla_simulator_score[1]))

        # simulator对数据中曝光的的pv-wise成交概率打分
        ground_truth_pv = state[:self.k*self.item_dim]
        ground_truth_pv = np.reshape(ground_truth_pv, (1, 28*6))
        label_simulator_score = self.simulator(torch.Tensor(ground_truth_pv)).detach().numpy()[0]
        label_simulator_score = np.exp(label_simulator_score[1]) / (np.exp(label_simulator_score[0]) + np.exp(label_simulator_score[1]))

        # select_index_in_state = [] # action中选中的动作对应候选中的index
        # for i in range(self.k):
        #     lla_action_i = lla_action[i * self.item_dim: (i + 1) * self.item_dim]
        #     for i in range(self.k):
        #         state_i = state[i * self.item_dim: (i + 1) * self.item_dim]
        #         if sum(lla_action_i) == sum(state_i):
        #             select_index_in_state.append(i)
        #             break
        #
        # # 选中的action与数据pv的重合度
        # similarity_score = len(select_index_in_state)

        # discount cumulative LLA reward
        lla_discount_reward = 0
        for i, lla_reward_i in enumerate(lla_reward_list):
            lla_discount_reward += lla_reward_i * (self.gamma ** i)

        # LLA's gmv
        # lla_gmv = lla_simulator_score.data * self.sum_pv_price / self.k

        # TODO: 组织5种属性的得分
        reward = lla_simulator_score + lla_discount_reward

        if mode == 'test':
            return lla_simulator_score
            # return float(lla_gmv.data)

        return reward

    def reset(self):
        '''

        :return: initial observation(state): item*16, label: 3(曝光、点击、购买)*16
        '''
        observation = self.dataset[self.data_pointer]
        self.data_pointer += 1
        state, label = self.to_state(copy.deepcopy(observation))
        self.sum_pv_price = 0

        return state, label

    def step(self, state, goal, action, lla_reward_list, mode='train'):
        '''
        @:param: state: HLA的当前状态
        @:param: goal: HLA生成的proto-pv
        @:param: action: LLA返回组织好的PV，从observation的16个候选中挑选出的6个item
        @:param: lla_reward: LLA返回的累积reward_list, len(reward_list)=6
        :return: state_{t+1}, reward, done
        '''

        observation = self.dataset[self.data_pointer]
        state_, label = self.to_state(copy.deepcopy(observation))

        reward = self.to_reward(state=state, label=label, goal=goal, lla_action=action, lla_reward_list=lla_reward_list, mode=mode)

        done = True if random.random() < self.random_done_prop else False # 随机停止，模拟一个用户停止浏览
        return state_, label, reward, done

# lla 16选6场景建模
class Env_low_level(object):
    '''
    state [item_feature*16, goal, position_vec]
    action 0~5
    reward ndcg@6
    state_ [item_feature*16, goal, position_vec[action:action+1]=action_feature]
    '''
    def __init__(self):
        self.state = ...
        self.label = ...
        self.goal = ...

        self.lamda = 0.5 # trade-off click and pay
        self.delta = 10 # clip purchase reward (e^10 = 22026)
        self.data_pointer = 0 # 数据集指针
        self.k = 6
        self.canditate_size = 16
        self.item_dim = 28
        self.select_cnt = 0 # 记录已经选了多少个

        self.last_gmv = 0 # 辅助奖励做差
        self.last_ndcg = 0
        self.total_reward = 0 # 记录两项奖励之和(不差分)
        self.reward_alpha = 1 # reward = å ndcg + (1-å) gmv

        self.select_vec = ...       # 已选已放的action_vector
        self.select_index = ...     # 已选已放的action_index
        self.canditate_set = []     # 候选动作空间

        self.dataset = dp.load_hierachical_data(filename=file_path)
        # self.dataset = dp.load_criteo_far_data()
        self.simulator = load_simulator()

    def reset(self, state, label, goal, pointer=0):
        '''

        :param state: 从HLA观察到的observation, item*16
        :param label: 从HLA观察到的item*16对应的label
        :param goal:  HLA's proto-pv
        :param pointer: 当前的数据指针
        :return: 初始状态(state, goal, select_vec) 28*16+28*6+28*6
        '''
        self.state = state.astype('float64')
        self.canditate_set = state.copy()
        self.label = label
        self.goal = goal
        self.data_pointer = pointer
        self.select_vec = np.full((self.item_dim * self.k,), -1.0)
        self.select_index = []
        self.select_cnt = 0

        self.last_gmv = 0 # 辅助奖励做差
        self.last_ndcg = 0

        tmp = np.append(self.state, self.goal)
        tmp1 = np.append(tmp, self.select_vec)
        return tmp1

    def step(self, action_index, pointer):
        '''

        :param state: 状态(state, goal, select_vec)
        :param action_index: 16个候选item中选中的item对应的index
        :param pointer: 矫正数据集指针
        :return: state_{t+1}, reward, done
        '''
        state = self.state
        action_vec = state[self.item_dim * action_index: self.item_dim * (action_index + 1)].copy()
        state[self.item_dim * action_index: self.item_dim * (action_index + 1)] = [-1] * self.item_dim

        select_vec = self.select_vec
        select_vec[self.item_dim * self.select_cnt: self.item_dim * (self.select_cnt + 1)] = action_vec
        self.select_index.append(action_index)
        self.select_cnt += 1

        self.state = state
        self.select_vec = select_vec
        self.canditate_set = state.copy()

        tmp = np.append(self.state, self.goal)
        tmp1 = np.append(tmp, self.select_vec)
        state_ = tmp1
        done = True if self.select_cnt == 6 else False

        reward = self.to_reward(position_vec=select_vec, action_list=self.select_index)

        return state_, reward, done

    def to_reward(self, position_vec, action_list, action='', dynamic=True):
        if dynamic:
            reward_ndcg_new = self.to_ndcg_2D(position_vec, action) # 动态NDCG
        else:
            reward_ndcg_new = self.to_ndcg_ori_rank(position_vec, action) # 静态NDCG

        if self.select_cnt == 6:
            self.total_reward = reward_ndcg_new

        reward_ndcg = reward_ndcg_new - self.last_ndcg
        self.last_ndcg = reward_ndcg_new

        # 差值reward
        reward = reward_ndcg

        return reward

    def to_gmv(self, action_list, position_vec):
        reward_gmv = 0
        for i, action_index in enumerate(action_list):
            _, click, purchase = self.label[action_index]
            price = position_vec[i*self.item_dim: (i+1)*self.item_dim][-5]
            reward_gmv += self.lamda * click + (1 - self.lamda) * purchase * min(math.log(1 + price), self.delta)

        return reward_gmv

    def to_ndcg_2D(self, position_vec, action=''):
        '''
        六宫格视的rank序视为purchase=1:=rank=1, click=1:=rank=2...
        :param position_vec: 当前哪些位置防止了哪些item
        :return: 以pv内原始序为ground truth前提下的ndcg
        '''
        pv = self.dataset[self.data_pointer]
        # debug_list = []
        # for p in pv:
        #     debug_list.append(p['feature'])
        # debug_list = np.array(debug_list).astype(np.float)
        item_list = []
        for i in range(self.k):
            item = item_vector_match_reward()
            item.original_rank = i
            item.item_feature = np.array(pv[i]['feature']).astype(np.float)
            item.sum_key = np.sum(item.item_feature)
            rel_i = float(pv[i]['rel'])
            item.rel = rel_i

            item_list.append(item)

        # 按照item.rel反向排序后下标即为新序
        item_list = sorted(item_list, key=lambda item: item.rel, reverse=True)

        IDCG = 0
        for i in range(self.k):
            item = item_list[i]
            rel = item.rel
            item.new_rank = i
            IDCG += (2.0**rel-1.0)/ (math.log2(item.new_rank+2.0))

        DCG = 0
        for i in range(self.k):
            action_i = position_vec[i*self.item_dim:(i+1)*self.item_dim]
            # action_i = np.array(pv[i]['feature']).astype(np.float)
            if -1 in action_i: continue
            action_i_sum_key = np.sum(action_i)

            # 在item_list(即goal)中寻找是否能匹配到
            match = False
            match_index = -1
            match_rel = 0
            for j in range(self.k):
                ideal_item = item_list[j]
                if i == ideal_item.original_rank:
                    match_index = ideal_item.new_rank
                if action_i_sum_key == ideal_item.sum_key:
                    match = True
                    match_rel = ideal_item.rel

            if match is False: continue
            # 如果匹配到，则换算过来实际位置i应该对应的重排位置index
            index = match_index
            rel = match_rel
            DCG += (2.0**rel-1) / (math.log2(index+2.0))

        ans = DCG/IDCG
        return ans

    def to_ndcg_ori_rank(self, position_vec, action=''):
        '''
        六宫格视的rank序视为1、2、3、4、5、6
        :param position_vec: 当前哪些位置防止了哪些item
        :return: 以pv内原始序为ground truth前提下的ndcg
        '''
        pv = self.dataset[self.data_pointer]
        # debug_list = []
        # for p in pv:
        #     debug_list.append(p['feature'])
        # debug_list = np.array(debug_list).astype(np.float)
        item_list = []
        for i in range(self.k):
            item = item_vector_match_reward()
            item.item_feature = np.array(pv[i]['feature']).astype(np.float)
            item.sum_key = np.sum(item.item_feature)
            e = pv[i]['exposure']
            c = pv[i]['click']
            p = pv[i]['purchase']
            rel_i = 0
            if p == '1':
                rel_i = 3
            elif c == '1':
                rel_i = 2
            elif e == '1':
                rel_i = 1
            item.rel = rel_i

            item_list.append(item)

        IDCG = 0
        for i in range(self.k):
            item = item_list[i]
            rel = item.rel
            IDCG += (2.0**rel-1.0)/ (math.log2(i+2.0))

        DCG = 0
        for i in range(self.k):
            action_i = position_vec[i*self.item_dim:(i+1)*self.item_dim]
            # action_i = np.array(pv[i]['feature']).astype(np.float)
            if -1 in action_i: continue
            action_i_sum_key = np.sum(action_i)

            # 在item_list(即goal)中寻找是否能匹配到
            match = False
            match_rel = 0
            for j in range(self.k):
                ideal_item = item_list[j]
                if action_i_sum_key == ideal_item.sum_key:
                    match = True
                    match_rel = ideal_item.rel
                    break

            if match is False: continue
            # 如果匹配到，则匹配到对应的rel
            rel = match_rel
            DCG += (2.0**rel-1) / (math.log2(i+2.0))

        ans = DCG/IDCG
        return ans

    def cosine_similarity(self, x, y):
        num = x.dot(y.T)
        denom = np.linalg.norm(x) * np.linalg.norm(y) + 1e-6
        return num / denom

    def match_action_cluster(self, action_vec, cluster, name='kmeans'):
        Action_clus = []
        if name == 'dbscan':
            action = cluster.fit_predict(np.reshape(action_vec,(1,-1)))
        if name == 'kmeans':
            action = cluster.predict(np.reshape(action_vec, (1, -1)))
        # find action_vec belong to which cluster
        for i in range(self.canditate_size):
            action_i = self.canditate_set[self.item_dim * i: self.item_dim * (i + 1)].astype(np.float)
            if -1 in action_i or np.sum(action_i)==0:
                continue
            if name == 'dbscan':
                clu_i = cluster.fit_predict(np.reshape(action_i,(1,-1)))
            if name == 'kmeans':
                clu_i = cluster.predict(np.reshape(action_i, (1, -1)))
            if clu_i == action:
                Action_clus.append(i)

        # find action belong to which action_index
        max_cos = -1
        max_index = -1
        for i in range(len(Action_clus)):
            index = Action_clus[i]
            action_index = self.canditate_set[self.item_dim * index: self.item_dim * (index + 1)].astype(np.float)
            cos = self.cosine_similarity(action_vec, action_index)
            if cos > max_cos:
                max_cos = cos
                max_index = index

        if max_index == -1:
            max_index = np.random.randint(0,self.canditate_size)

        return max_index

class Env_high_level_v3(object):
    def __init__(self):
        self.data_pointer = 0
        self.dataset = dp.load_hierachical_data()
        # self.dataset = dp.load_data()

        self.state_one_line = []
        self.label_one_line = []

        self.k = 6
        self.canditate_size = 16
        self.random_done_prop = 0.1
        self.item_dim = 28
        self.gamma = 0.9 # reward discount

        self.sum_pv_price = 0
        self.simulator = load_simulator()

    def to_state(self, data_piece):
        X_t = []
        for item in data_piece:
            X_t.extend(item['feature'])

        state = []
        label = []
        for item in data_piece:
            q = item['feature']
            q.extend(X_t)
            state.append(q)

            label.append([eval(item['rel']),eval(item['click']),eval(item['purchase'])])

        # 返回一条包含的所有state（包含16条数据）16 * 476(28*16+16), 其对应的标签label 16 * 3
        # 返回一条包含的所有X_t（包含16条数据）16 * 448(28*16), 其对应的标签label 16 * 3
        self.state_one_line = X_t
        self.label_one_line = label

        X_t = np.array(X_t).astype('float64')
        # np.reshape(state, (-1, state_dim))
        return X_t, label

    def to_reward(self, state, label, lla_action, lla_reward_list, mode='train'):
        '''

        :param state: HLA的16个候选item 28*16
        :param label: 16个候选item对应的标签 3*16
        :param goal: HLA给出的proto-pv 28*6
        :param lla_action: LLA组织好的PV 28*6
        :param lla_reward_list: LLA返回的reward列表 1*6
        :return: simulator打分（+）pv与ground truth相似度（+）simulator(action_pv)/simulator(label_pv)
        '''

        # simulator对lla-action的pv-wise成交概率打分
        lla_action = np.reshape(lla_action, (1, 28*6))
        lla_simulator_score = self.simulator(torch.Tensor(lla_action)).detach().numpy()[0]
        lla_simulator_score = np.exp(lla_simulator_score[1]) / (np.exp(lla_simulator_score[0]) + np.exp(lla_simulator_score[1]))

        # discount cumulative LLA reward
        lla_discount_reward = 0
        for i, lla_reward_i in enumerate(lla_reward_list):
            lla_discount_reward += lla_reward_i * (self.gamma ** i)

        # TODO: 组织5种属性的得分
        reward = lla_simulator_score + lla_discount_reward # + hla_simulator_score

        if mode == 'test':
            return float(lla_simulator_score)

        return reward

    def reset(self):
        '''

        :return: initial observation(state): item*16, label: 3(曝光、点击、购买)*16
        '''
        observation = self.dataset[self.data_pointer]
        self.data_pointer += 1
        state, label = self.to_state(copy.deepcopy(observation))
        self.sum_pv_price = 0

        return state, label

    def step(self, state, action, lla_reward_list, mode='train'):
        '''
        @:param: state: HLA的当前状态
        @:param: action: LLA返回组织好的PV，从observation的16个候选中挑选出的6个item
        @:param: lla_reward: LLA返回的累积reward_list, len(reward_list)=6
        :return: state_{t+1}, reward, done
        '''

        observation = self.dataset[self.data_pointer]
        state_, label = self.to_state(copy.deepcopy(observation))

        reward = self.to_reward(state=state, label=label, lla_action=action, lla_reward_list=lla_reward_list, mode=mode)

        done = True if random.random() < self.random_done_prop else False # 随机停止，模拟一个用户停止浏览
        return state_, label, reward, done

# lla 16选6场景建模
class Env_low_level_v3(object):
    '''
    state [item_feature*16, position_vec]
    action 0~5
    reward ndcg@6
    state_ [item_feature*16, position_vec[action:action+1]=action_feature]
    '''
    def __init__(self):
        self.state = ...
        self.label = ...
        self.goal = ...

        self.lamda = 0.5 # trade-off click and pay
        self.delta = 10 # clip purchase reward (e^10 = 22026)
        self.data_pointer = 0 # 数据集指针
        self.k = 6
        self.canditate_size = 16
        self.item_dim = 28
        self.select_cnt = 0 # 记录已经选了多少个

        self.last_gmv = 0 # 辅助奖励做差
        self.last_ndcg = 0
        self.total_reward = 0 # 记录两项奖励之和(不差分)
        self.reward_alpha = c_reward_alpha # reward = å ndcg + (1-å) gmv

        self.select_vec = ...       # 已选已放的action_vector
        self.select_index = ...     # 已选已放的action_index
        self.canditate_set = []     # 候选动作空间

        self.dataset = dp.load_hierachical_data()

    def reset(self, state, label, preserve_cnt, pointer):
        '''

        :param state: 从HLA观察到的observation, item*16
        :param label: 从HLA观察到的item*16对应的label
        :param preserve_cnt:  前cnt个保持原序, 0~5
        :param pointer: 当前的数据指针
        :return: 初始状态(state, select_vec) 28*16+28*6
        '''
        self.state = state.astype('float64')
        self.canditate_set = state.copy()
        self.label = label
        self.preserve_cnt = preserve_cnt
        self.data_pointer = pointer

        self.select_vec = np.full((self.item_dim * self.k,), -1.0)
        self.select_cnt = 0
        self.select_index = []
        if preserve_cnt > 0:
            for i in range(preserve_cnt):
                self.step(i, pointer)

        self.last_gmv = 0 # 辅助奖励做差
        self.last_ndcg = 0

        return np.append(self.state, self.select_vec)

    def step(self, action_index, pointer):
        '''

        :param state: 状态(state, goal, select_vec)
        :param action_index: 16个候选item中选中的item对应的index
        :param pointer: 矫正数据集指针
        :return: state_{t+1}, reward, done
        '''
        state = self.state
        action_vec = state[self.item_dim * action_index: self.item_dim * (action_index + 1)].copy()
        state[self.item_dim * action_index: self.item_dim * (action_index + 1)] = [-1] * self.item_dim

        select_vec = self.select_vec
        select_vec[self.item_dim * self.select_cnt: self.item_dim * (self.select_cnt + 1)] = action_vec
        self.select_index.append(action_index)
        self.select_cnt += 1

        self.state = state
        self.select_vec = select_vec
        self.canditate_set = state.copy()

        state_ = np.append(self.state, self.select_vec)
        done = True if self.select_cnt == 6 else False

        reward = self.to_reward(position_vec=select_vec, action_list=self.select_index)

        return state_, reward, done

    def to_reward(self, position_vec, action_list, action='', dynamic=True):
        if dynamic:
            reward_ndcg_new = self.to_ndcg_2D(position_vec, action) # 动态NDCG
        else:
            reward_ndcg_new = self.to_ndcg_ori_rank(position_vec, action) # 静态NDCG

        if self.select_cnt == 6:
            self.total_reward = reward_ndcg_new

        reward_ndcg = reward_ndcg_new - self.last_ndcg
        self.last_ndcg = reward_ndcg_new

        # 差值reward
        reward = reward_ndcg

        return reward

    def to_ndcg_2D(self, position_vec, action=''):
        '''
        六宫格视的rank序视为purchase=1:=rank=1, click=1:=rank=2...
        :param position_vec: 当前哪些位置防止了哪些item
        :return: 以pv内原始序为ground truth前提下的ndcg
        '''
        pv = self.dataset[self.data_pointer]
        # debug_list = []
        # for p in pv:
        #     debug_list.append(p['feature'])
        # debug_list = np.array(debug_list).astype(np.float)
        item_list = []
        for i in range(self.k):
            item = item_vector_match_reward()
            item.original_rank = i
            item.item_feature = np.array(pv[i]['feature']).astype(np.float)
            item.sum_key = np.sum(item.item_feature)
            rel_i = float(pv[i]['rel'])
            item.rel = rel_i

            item_list.append(item)

        # 按照item.rel反向排序后下标即为新序
        item_list = sorted(item_list, key=lambda item: item.rel, reverse=True)

        IDCG = 0
        for i in range(self.k):
            item = item_list[i]
            rel = item.rel
            item.new_rank = i
            IDCG += (2.0 ** rel - 1.0) / (math.log2(item.new_rank + 2.0))

        DCG = 0
        for i in range(self.k):
            action_i = position_vec[i * self.item_dim:(i + 1) * self.item_dim]
            # action_i = np.array(pv[i]['feature']).astype(np.float)
            if -1 in action_i: continue
            action_i_sum_key = np.sum(action_i)

            # 在item_list(即goal)中寻找是否能匹配到
            match = False
            match_index = -1
            match_rel = 0
            for j in range(self.k):
                ideal_item = item_list[j]
                if i == ideal_item.original_rank:
                    match_index = ideal_item.new_rank
                if action_i_sum_key == ideal_item.sum_key:
                    match = True
                    match_rel = ideal_item.rel

            if match is False: continue
            # 如果匹配到，则换算过来实际位置i应该对应的重排位置index
            index = match_index
            rel = match_rel
            DCG += (2.0 ** rel - 1) / (math.log2(index + 2.0))

        ans = DCG / IDCG
        return ans

    def to_ndcg_ori_rank(self, position_vec, action=''):
        '''
        六宫格视的rank序视为1、2、3、4、5、6
        :param position_vec: 当前哪些位置防止了哪些item
        :return: 以pv内原始序为ground truth前提下的ndcg
        '''
        pv = self.dataset[self.data_pointer]
        # debug_list = []
        # for p in pv:
        #     debug_list.append(p['feature'])
        # debug_list = np.array(debug_list).astype(np.float)
        item_list = []
        for i in range(self.k):
            item = item_vector_match_reward()
            item.item_feature = np.array(pv[i]['feature']).astype(np.float)
            item.sum_key = np.sum(item.item_feature)
            e = pv[i]['exposure']
            c = pv[i]['click']
            p = pv[i]['purchase']
            rel_i = 0
            if p == '1':
                rel_i = 3
            elif c == '1':
                rel_i = 2
            elif e == '1':
                rel_i = 1
            item.rel = rel_i

            item_list.append(item)

        IDCG = 0.0001
        for i in range(self.k):
            item = item_list[i]
            rel = item.rel
            IDCG += (2.0**rel-1.0)/ (math.log2(i+2.0))

        DCG = 0
        for i in range(self.k):
            action_i = position_vec[i*self.item_dim:(i+1)*self.item_dim]
            # action_i = np.array(pv[i]['feature']).astype(np.float)
            if -1 in action_i: continue
            action_i_sum_key = np.sum(action_i)

            # 在item_list(即goal)中寻找是否能匹配到
            match = False
            match_rel = 0
            for j in range(self.k):
                ideal_item = item_list[j]
                if action_i_sum_key == ideal_item.sum_key:
                    match = True
                    match_rel = ideal_item.rel
                    break

            if match is False: continue
            # 如果匹配到，则匹配到对应的rel
            rel = match_rel
            DCG += (2.0**rel-1) / (math.log2(i+2.0))

        ans = DCG/IDCG
        return ans

    def cosine_similarity(self, x, y):
        num = x.dot(y.T)
        denom = np.linalg.norm(x) * np.linalg.norm(y) + 1e-6
        return num / denom

    def match_action_cluster(self, action_vec, cluster, name='kmeans'):
        Action_clus = []
        if name == 'dbscan':
            action = cluster.fit_predict(np.reshape(action_vec,(1,-1)))
        if name == 'kmeans':
            action = cluster.predict(np.reshape(action_vec, (1, -1)))
        # find action_vec belong to which cluster
        for i in range(self.canditate_size):
            action_i = self.canditate_set[self.item_dim * i: self.item_dim * (i + 1)].astype(np.float)
            if -1 in action_i or np.sum(action_i)==0:
                continue
            if name == 'dbscan':
                clu_i = cluster.fit_predict(np.reshape(action_i,(1,-1)))
            if name == 'kmeans':
                clu_i = cluster.predict(np.reshape(action_i, (1, -1)))
            if clu_i == action:
                Action_clus.append(i)

        # find action belong to which action_index
        max_cos = -1
        max_index = -1
        for i in range(len(Action_clus)):
            index = Action_clus[i]
            action_index = self.canditate_set[self.item_dim * index: self.item_dim * (index + 1)].astype(np.float)
            cos = self.cosine_similarity(action_vec, action_index)
            if cos > max_cos:
                max_cos = cos
                max_index = index

        if max_index == -1:
            max_index = np.random.randint(0,self.canditate_size)

        return max_index

    def match_action_cos(self, action_vec):
        state = self.canditate_set
        Action_prob = np.zeros((self.k,))
        for i in range(self.k):
            action_i = state[self.item_dim * i: self.item_dim * (i + 1)].astype(np.float)
            if -1 in action_i:
                continue
            cos = self.cosine_similarity(action_vec, action_i)
            Action_prob[i] = cos
        return np.argmax(Action_prob)

# 测试上游pv数据的表现
class Env_upstream(object):
    def __init__(self):
        self.k = 6
        self.candidate_size = 16
        self.item_dim = 28

        self.data_pointer = 0
        self.dataset = dp.load_hierachical_data(filename=file_path)
        # self.dataset = dp.load_criteo_far_data()

        self.simulator = load_simulator()

    def to_state(self, data_piece):
        X_t = []
        for item in data_piece:
            X_t.extend(item['feature'])

        state = []
        label = []
        for item in data_piece:
            q = item['feature']
            q.extend(X_t)
            state.append(q)

            label.append([eval(item['rel']),eval(item['click']),eval(item['purchase'])])

        # 返回一条包含的所有state（包含16条数据）16 * 476(28*16+16), 其对应的标签label 16 * 3
        # 返回一条包含的所有X_t（包含16条数据）16 * 448(28*16), 其对应的标签label 16 * 3
        self.state_one_line = X_t
        self.label_one_line = label

        X_t = np.array(X_t)
        # np.reshape(state, (-1, state_dim))
        return X_t, label

    def reset(self):
        '''

        :return: initial observation(state): item*16, label: 3(曝光、点击、购买)*16
        '''
        observation = self.dataset[self.data_pointer]
        self.data_pointer += 1
        state, label = self.to_state(copy.deepcopy(observation))

        return state, label

    def to_reward(self, state, reward_mode='simulator_score', pv_mode='logpv'):
        '''

        :param state:
        :param reward_mode: 返回gmv或simulator的打分
        :param pv_mode: 使用log中原始的pv，或使用混乱到的pv(用来验证simulator)
        :return:
        '''

        if pv_mode == 'logpv': # 不作处理
            action = state[:self.k*self.item_dim].astype('float64')
        elif pv_mode == 'last2firstpv': # 把最后一个放在第一位
            action = np.append(state[-self.item_dim:], state[self.item_dim: self.k * self.item_dim]).astype('float64')
        elif pv_mode == 'sfarnkpv': # 把第一个和第六个换位置
            first = state[0:84]
            last = state[84: 168]
            # action = np.append(last, first)
            action = np.append(last,first).astype('float64')
        price = self.to_price(action)

        # simulator对lla-action的pv-wise成交概率打分
        simulator_score = self.simulator(torch.Tensor(action))
        simulator_score = np.exp(simulator_score[1]) / (np.exp(simulator_score[0]) + np.exp(simulator_score[1]))

        # pv价格成交量期望
        reward = 0
        if reward_mode == 'gmv':
            reward = simulator_score * price / self.k
        elif reward_mode == 'simulator_score':
            reward = simulator_score

        return float(reward.data)

    def to_price(self, state):
        sum_price = 0
        for i in range(self.k):
            item_i = state[i*self.item_dim: (i+1)*self.item_dim].astype('float64')
            sum_price += item_i[-5]

        return sum_price






