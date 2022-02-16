from open_data.config import *
import pandas as pd
import numpy as np
import math
import random
random.seed(1)
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


'''
先给一个大致的数据介绍，
一行（图中的一个方框）表示一个pv的数据，
一行数据里面会包含16个商品的特征与标签，
每个商品之间用分号；间隔，
同个商品的特征与标签用逗号分隔，
每个商品的前3维分别表示曝光、点击、购买标签
（正常情况下，一个pv内16个商品只有前6个商品的曝光标签是1，后10个商品的曝光标签为0），
每个商品的后28维均表示商品特征
'''
def load_data(filename='./data/20210319.txt', size=data_size):
    dataset = []

    f = open(filename, 'r')
    cnt = 0
    print('Loading Data...')
    for _ in trange(size):
        data_pv = []
        pv = f.readline().strip('\n')
        if len(pv) == 0 or cnt == size:
            break

        pv = pv.split(';')
        for item in pv:
            item = item.split(',')
            data_item = {}
            data_item['exposure'] = item[0]
            data_item['click'] = item[1]
            data_item['purchase'] = item[2]
            data_item['feature'] = item[3:]

            data_pv.append(data_item)

        dataset.append(np.array(data_pv))
        cnt += 1

    print('Finish loading data...')
    # return np.array(dataset)
    return dataset

def load_normalized_data(filename='./data/20210319.txt', size=data_size):
    dataset = []

    f = open(filename, 'r')
    cnt = 0
    print("Loading Data...")
    for _ in trange(size):
        data_pv = []
        pv = f.readline().strip('\n')
        if len(pv) == 0 or cnt == size:
            break

        pv = pv.split(';')
        max_feature = np.zeros(action_dim)
        for item in pv:
            item = item.split(',')
            data_item = {}
            data_item['exposure'] = item[0]
            data_item['click'] = item[1]
            data_item['purchase'] = item[2]
            data_item['feature'] = item[3:]
            for feature_index in range(action_dim):
                feature_i = eval(data_item['feature'][feature_index])
                if feature_i > max_feature[feature_index]:
                    max_feature[feature_index] = feature_i
            data_pv.append(data_item)

        # 每个pv按最大值归一化
        for pv_index in range(len(data_pv)):
            for feature_index in range(action_dim):
                data_pv[pv_index]['feature'][feature_index] = eval(data_pv[pv_index]['feature'][feature_index]) / (max_feature[feature_index] + 1e-6)

        dataset.append(np.array(data_pv))
        cnt += 1

    # return np.array(dataset)
    debug = np.array(dataset)
    return dataset

def load_logprice_data(filename='./data/20210319.txt', size=data_size):
    dataset = []

    f = open(filename, 'r')
    cnt = 0
    print("Loading Data...")
    for _ in trange(size):
        data_pv = []
        pv = f.readline().strip('\n')
        if len(pv) == 0 or cnt == size:
            break

        pv = pv.split(';')
        max_feature = np.zeros(action_dim)
        for item in pv:
            item = item.split(',')
            data_item = {}
            data_item['exposure'] = item[0]
            data_item['click'] = item[1]
            data_item['purchase'] = item[2]
            data_item['feature'] = item[3:]

            data_item['feature'][-5] = math.log(eval(data_item['feature'][-5])) # log price

            data_pv.append(data_item)

        dataset.append(np.array(data_pv))
        cnt += 1

    # return np.array(dataset)
    debug = np.array(dataset)
    return dataset

# 伪造高斯分布模拟未点击数据
def load_hierachical_data(filename='./data/criteo_pv_rec_normalFill.txt', size=data_size):
    dataset = []

    f = open(filename, 'r')
    cnt = 0
    print("Loading Data...")
    for _ in trange(size):
        data_pv = []
        pv = f.readline().strip('\n')
        if len(pv) == 0 or cnt == size:
            break
        '''
            pv的结构为：rel, click, purchase(conversion), 28feature, rel, click, purchase(conversion), 28feature,...(16个)
        '''
        pv = pv.split(',')
        for i in range(16):
            item = pv[i*31: (i+1)*31]
            data_item = {}
            data_item['rel'] = item[0]
            data_item['click'] = item[1]
            data_item['purchase'] = item[2]
            data_item['feature'] = item[3:]

            data_pv.append(data_item)

        dataset.append(np.array(data_pv))
        cnt += 1

    # return np.array(dataset)
    debug = np.array(dataset)
    return dataset

# 混乱分层时序数据
def shuffle_hierachical_data(filename='./data/h_20210416.txt', size=data_size, SHUFFLE=True):
    dataset = []

    f = open(filename, 'r')
    cnt = 0
    print("Loading Data...")
    for _ in trange(size):
        pv = f.readline().strip('\n')
        dataset.append(pv)

    # return np.array(dataset)
    if SHUFFLE is True:
        random.shuffle(dataset)
    debug = np.array(dataset)

    np.savetxt('./data/h_20210416_shuffle.txt',debug, fmt='%s')

    return dataset

# 远距离点击数据模拟未点击数据
def load_criteo_far_data(size=5000):
    file1 = './data/criteo_pv_rec_farFill_part1.txt'
    file2 = './data/criteo_pv_rec_farFill_part2.txt'
    f1 = open(file1, 'r')
    f2 = open(file2, 'r')

    cnt = 0
    print("Loading Data...")
    dataset = []
    f = f1
    for i in trange(2*size):
        data_pv = []
        pv = f.readline().strip('\n')
        if cnt == size-1 or len(pv) == 0:
            f = f2
        elif cnt == 2*size-1 and len(pv) == 0:
            break
        '''
            pv的结构为：rel, click, purchase(conversion), 28feature, rel, click, purchase(conversion), 28feature,...(16个)
        '''
        pv = pv.split(',')
        for i in range(16):
            item = pv[i*31: (i+1)*31]
            data_item = {}
            data_item['rel'] = item[0]
            data_item['click'] = item[1]
            data_item['purchase'] = item[2]
            data_item['feature'] = item[3:]

            data_pv.append(data_item)

        dataset.append(np.array(data_pv))
        cnt += 1

    # return np.array(dataset)
    debug = np.array(dataset)
    return dataset

# 提取数据的动作特征
def generate_action_feature():
    data = load_data()
    # data = load_normalized_data()
    # data = load_logprice_data()
    action_feature = []
    for pv in data:
        for index in range(canditate_size):
            action_feature.append(pv[index]['feature'])

    return np.array(action_feature)

#  按pv对每一维进行归一化
def generate_normalized_action_feature():
    data = load_data()
    action_feature = []
    max_feature = np.zeros(action_dim)
    for pv in data:
        for index in range(canditate_size):
            for feature_index in range(action_dim):
                feature_i = eval(pv[index]['feature'][feature_index])
                if feature_i > max_feature[feature_index]:
                    max_feature[feature_index] = feature_i

        for index in range(canditate_size):
            for feature_index in range(action_dim):
                pv[index]['feature'][feature_index] = eval(pv[index]['feature'][feature_index]) / (max_feature[feature_index] + 1e-6)
            action_feature.append(pv[index]['feature'])

        # for index in range(canditate_size):
        #     a_index = pv[index]['feature']
        #     a_index[23] = eval(a_index[23])/max_price
        #     action_feature.append(a_index)

    return np.array(action_feature)

def extract_action_feature_from_memory(memory):
    ms = memory[:, :state_dim]
    action_feature = []
    for state in ms:
        if -1 in state: continue
        for index in range(canditate_size):
            feature_i = state[index*action_dim: (index+1)*action_dim]
            if -1 in feature_i: continue
            action_feature.append(feature_i)

    return action_feature

# 从LLA memory中抽取action特征
def extract_action_feature_from_lla_memory(memory):
    ms = memory[:, :state_dim]
    action_feature = []
    for lla_state in ms:
        state = lla_state[:item_dim*canditate_size]
        if -1 in state: continue
        for index in range(canditate_size):
            feature_i = state[index*item_dim: (index+1)*item_dim]
            if -1 in feature_i: continue
            action_feature.append(feature_i)

    return action_feature

# 提取分层数据的动作特征
def generate_hierachical_action_feature():
    data = load_hierachical_data()

    action_feature = []
    for pv in data:
        for index in range(canditate_size):
            action_feature.append(pv[index]['feature'])

    return np.array(action_feature)

# 统计分层数据价格特征
def count_hierachical_price():
    data = load_hierachical_data()

    action_feature = []
    price_feature = []
    price_dict = {}

    for pv in data:
        max = 0
        min = 1e8
        for index in range(canditate_size):
            price_i = eval(pv[index]['feature'][23])
            if price_i > max:
                max = price_i
            if price_i < min:
                min = price_i
            # action_feature.append(pv[index]['feature'])

            price_feature.append(price_i)
        # price_feature.append(max - min)

    # return np.array(action_feature)
    return sorted(price_feature)

# 统计分层数据的点击和购买行为分布
def count_hierachical_click_and_purchase(filename='./data/h_20210416.txt', size=data_size):
    dataset = []
    click_cnt = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    purchase_cnt = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    f = open(filename, 'r')
    cnt = 0
    for _ in range(size):
        data_pv = []
        pv = f.readline().strip('\n')
        if len(pv) == 0 or cnt == size:
            break
        '''
            pv的结构为：user# time# trigger_bigcate_ID# trigger_smallcate_ID# original_pv
        '''
        pv_ = pv.split('#')
        pv = pv_[-1].split(';')

        item_i = 0
        for item in pv:
            item = item.split(',')
            if item[1] == '1':
                click_cnt[item_i] += 1
            if item[2] == '1':
                purchase_cnt[item_i] += 1
            item_i += 1

            if item_i >= 6:
                break

        # dataset.append(np.array(data_pv))
        cnt += 1

    # return np.array(dataset)
    # debug = np.array(dataset)
    cvr = []
    for i in range(6):
        c = click_cnt[i]
        v = purchase_cnt[i]
        cvr.append(v / c)
    return click_cnt, purchase_cnt, cvr

# 统计分层数据的点击和购买行为分布
class user_info:
    def __init__(self, id):
        self.user_id = id
        self.total_num = 0
        self.click_cnt = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self.purchase_cnt = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

def count_hierachical_c_p_perperson(filename='./data/h_20210416.txt', size=data_size):
    dataset = []
    click_cnt = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    purchase_cnt = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    user_cnt = {}

    f = open(filename, 'r')
    cnt = 0
    for _ in range(size):
        data_pv = []
        pv = f.readline().strip('\n')
        if len(pv) == 0 or cnt == size:
            break
        '''
            pv的结构为：user# time# trigger_bigcate_ID# trigger_smallcate_ID# original_pv
        '''
        pv_ = pv.split('#')
        user_id = pv_[0]
        pv = pv_[-1].split(';')

        if user_id not in user_cnt:
            tmp = user_info(user_id)
            tmp.total_num = 1
            user_cnt[user_id] = tmp
        else:
            user_cnt[user_id].total_num += 1

        for item_index in range(6):
            item = pv[item_index].split(',')
            c = item[1]
            if c == '1':
                user_cnt[user_id].click_cnt[item_index] += 1
            p = item[2]
            if p == '1':
                user_cnt[user_id].purchase_cnt[item_index] += 1

    user_cnt_10 = []
    for user_id_key in user_cnt:
        if user_cnt[user_id_key].total_num > 10:
            user_cnt_10.append(user_cnt[user_id_key])

    user_cnt_10.sort(key=lambda x: x.total_num, reverse=True)
    return user_cnt_10

# for size in [5000, 10000, 20000]:
#     user_cnt = count_hierachical_c_p_perperson(size=size)
#     print(size, len(user_cnt))
# print(click_cnt)
# print(purchase_cnt)
