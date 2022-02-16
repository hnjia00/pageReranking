import numpy as np
import math
import random
random.seed(1)
from tqdm import trange
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.externals import joblib
from open_data.data_process import generate_action_feature, generate_hierachical_action_feature

def generate_kmeans_action_classfier(mode = 'h'):
    if mode == 'h':
        data = generate_hierachical_action_feature()
    else:
        data = generate_action_feature()

    action_classfier = KMeans(n_clusters=16) # 430

    action_classfier.fit(data)

    # print(action_classfier.labels_.tolist())

    joblib.dump(action_classfier, './data/action_classifier/kmeans/kmeans_hierachical_action_classfier.m')

def generate_DBSCAN_action_classfier():
    data = generate_action_feature()
    action_classfier = DBSCAN(eps=0.1, min_samples=20) # 不能指定聚类数目

    action_classfier.fit(data)

    # print(action_classfier.labels_.tolist())

    joblib.dump(action_classfier, './data/dbscan_action_classfier_eps0.1.m')


def generate_mini_batch_kmeans_action_classfier():
    data = generate_action_feature()
    action_classfier = MiniBatchKMeans(n_clusters=16, batch_size=2000) # 需要制定batch大小，更新时在训练数据中随机采batch个数据

    action_classfier.fit(data)

    # print(action_classfier.labels_.tolist())

    joblib.dump(action_classfier, './data/minikmeans_action_classfier.m')

# canopy预产生cluster_k的个数，如何适应batch 更新的场景？
class canopy_cluster(object):
    def __init__(self, P):
        self.center = np.array(P).astype(np.float)
        self.members = []

    def cal_distance(self, a_j):
        A = self.center
        B = np.array(a_j).astype(np.float)

        sum = 0.0
        for i in range(len(A)):
            sum += (A[i] - B[i])**2

        sum = math.sqrt(sum)
        # print(sum)
        return sum

def canopy_generate_k(action_feature_list, T1, T2):
    '''
    :param action_feature_list: 待划分的无标签动作特征
    :param T1: 阈值1 T1>T2
    :param T2: 阈值2
    :return: cluster的个数k
    '''
    action_feature_list = action_feature_list.tolist()
    cluster_centers = []
    while len(action_feature_list) != 0:
        index = random.randint(0, len(action_feature_list)-1)
        P = action_feature_list[index]

        center_P = canopy_cluster(P)
        cluster_centers.append(center_P)
        action_feature_list.pop(index)

        # 计算action_list中剩余样本到cluster_centers中每个类中心的距离
        pop_index = []
        # for cluster_index in range(len(cluster_centers)):
        #     center_P = cluster_centers[cluster_index]
        #     for action_index in range(len(action_feature_list)):
        #         action_feature_i = action_feature_list[action_index]
        #         dis = center_P.cal_distance(action_feature_i)
        #         # print(dis)
        #         if dis < T1:
        #             center_P.members.append(action_feature_i)
        #         if dis < T2:
        #             pop_index.append(action_index)
        #     for pop_i in range(len(pop_index)):
        #         action_feature_list.pop(pop_index[-pop_i-1])

        # 只比较新加入的cluster center
        center_P = cluster_centers[-1]
        for action_index in trange(len(action_feature_list)):
            action_feature_i = action_feature_list[action_index]
            dis = center_P.cal_distance(action_feature_i)
            # print(dis)
            if dis < T1:
                center_P.members.append(action_feature_i)
            if dis < T2:
                pop_index.append(action_index)
        for pop_i in range(len(pop_index)):
            action_feature_list.pop(pop_index[-pop_i - 1])

    k = len(cluster_centers)
    for center in cluster_centers:
        if len(center.members) < 50:
            k -= 1

    return k


generate_kmeans_action_classfier()
# generate_DBSCAN_action_classfier()
# generate_mini_batch_kmeans_action_classfier()

# data = generate_action_feature()
# k = canopy_generate_k(data, T1=2, T2=1.2)
# print('[final_k]',k) # k=342