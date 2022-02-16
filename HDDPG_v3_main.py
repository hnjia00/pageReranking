from open_data.config import *
from open_data.data_process import *
import tensorflow as tf
import numpy as np
import random
from open_data.HDDPG_Agent import meta_controller_DDPG, controller_DDPG
# from h_DDPG import meta_controller_DDPG, controller_DDPG
from open_data.Env_model import *
from sklearn.externals import joblib
from open_data.simulator.load_simulator import load_simulator
from open_data.DQN_Agent import higher_DQN
import torch
from tqdm import tqdm, trange

np.random.seed(1)
tf.set_random_seed(1)

sess = tf.Session()

# action_classifier = joblib.load('./data/action_classifier/dbscan/dbscan_action_classfier_eps0.1.m')
action_classifier = joblib.load('./data/action_classifier/kmeans/kmeans_hierachical_action_classfier.m')

# HLA pv预测场景建模
env_hla = Env_high_level_v3()
env_lla = Env_low_level_v3()

# pv-wise simulator
simulator = load_simulator()

def train_hDDPG_with_pvpredict(PRINT_train = False, PRINT_test = False):
    global EPSILON
    global c_EPSILON

    # pv predict
    high_level_agent = higher_DQN(6, state_dim)
    # 16 -> 6
    low_level_agent = controller_DDPG(c_action_dim, c_state_dim, c_action_bound)

    sess.run(tf.global_variables_initializer())

    sum_lla_reward_list = [0]
    sum_hla_reward_list = [0]
    action_feature_lla = []

    ########################## Train ##########################
    # 纯探索
    for i in trange(int(MAX_EPISODES * split_rate)):
        state_hla, label_canditate_16 = env_hla.reset()

        preserve_k = high_level_agent.choose_action(state_hla)

        # low level environment 16->6
        lla_reward_list = []
        state_lla = env_lla.reset(state=state_hla, label=label_canditate_16, preserve_cnt=preserve_k, pointer=i)
        for j in range(env_lla.k - preserve_k):
            while True:
                proto_item_feature = low_level_agent.choose_action(state_lla)
                proto_item_feature_index = env_lla.match_action_cluster(proto_item_feature, action_classifier)
                if random.random() < c_EPSILON:
                    c_EPSILON *= 0.995
                    proto_item_feature_index = random.randint(preserve_k, env_lla.canditate_size - 1)
                    # 随机选取动作时使用cluster类中心替代a
                    proto_item_feature = action_classifier.cluster_centers_[proto_item_feature_index]
                if proto_item_feature_index not in env_lla.select_index:
                    break

            state_lla_, reward_lla, done = env_lla.step(action_index=proto_item_feature_index, pointer=i)
            real_action_feature = state_hla[
                                  proto_item_feature_index * item_dim: (proto_item_feature_index + 1) * item_dim]
            low_level_agent.store_transition(state_lla, real_action_feature, reward_lla, state_lla_)
            lla_reward_list.append(reward_lla)
            if low_level_agent.pointer > c_MEMORY_CAPACITY:
                low_level_agent.learn()
                # incremental update action classifier
                if i > 1000 and low_level_agent.pointer % c_MEMORY_CAPACITY == 0 and INCRE_update:
                    action_feature = extract_action_feature_from_lla_memory(low_level_agent.memory)
                    action_feature_lla.extend(action_feature)
                    if len(action_feature_lla) % 3000 == 0 and len(action_feature) > 0:
                        action_classifier.fit(action_feature)
            if done:
                break

            state_lla = state_lla_

        state_hla_, label_, reward_hla, done = env_hla.step(state=state_hla, action=env_lla.select_vec,
                                                            lla_reward_list=lla_reward_list)

        high_level_agent.store_transition(s=state_hla, a=preserve_k, r=reward_hla, s_=state_hla_)

        if high_level_agent.memory_counter > MEMORY_CAPACITY:
            high_level_agent.learn()


    sum_lla_reward_list = [0]
    sum_hla_reward_list = [0]

    ########################## Test ##########################
    # high level environment proto-pv predict
    base_gmv_list = 0
    test_gmv_list = 0
    better = 0
    use_upstream = 0
    reward_list = []
    reward_sum = 0
    dynamic_ndcg = 0
    for i in trange(int(split_rate * MAX_EPISODES), MAX_EPISODES):
        state_hla, label_canditate_16 = env_hla.reset()

        upstream_pv = state_hla[:item_dim * env_hla.k]
        upstream_pv = np.reshape(upstream_pv, (1, 28*6))
        outputs = env_hla.simulator(torch.Tensor(upstream_pv)).detach().numpy()[0]
        upstream_score = np.exp(outputs[1]) / (np.exp(outputs[0])+np.exp(outputs[1]))

        preserve_k = high_level_agent.choose_action(state_hla)

        # low level environment 16->6
        lla_reward_list = []
        state_lla = env_lla.reset(state=state_hla, label=label_canditate_16, preserve_cnt=preserve_k, pointer=i)
        for j in range(env_lla.k-preserve_k):
            while True:
                proto_item_feature = low_level_agent.choose_action(state_lla)
                proto_item_feature_index = env_lla.match_action_cluster(proto_item_feature, action_classifier)
                # if random.random() < 0.05:
                #     proto_item_feature_index = random.randint(0, env_lla.canditate_size-1)

                if proto_item_feature_index not in env_lla.select_index:
                    break

            state_lla_, reward_lla, done = env_lla.step(action_index=proto_item_feature_index, pointer=i)

            lla_reward_list.append(reward_lla)  # 记录本轮lla奖励
            if done:
                sum_lla_reward = (env_lla.total_reward + sum_lla_reward_list[-1])*i / (i+1) # 使用total_reward记录lla排好后的总奖励，reward_lla为差分奖励
                sum_lla_reward_list.append(sum_lla_reward) # 记录平均lla奖励
                # if PRINT_test:
                #     print('[LLA average reward]:', sum_lla_reward)
                break

            state_lla = state_lla_

        if i == MAX_EPISODES - 1: break
        state_hla_, label_, reward_hla, done = env_hla.step(state=state_hla, action=env_lla.select_vec, lla_reward_list=lla_reward_list, mode='test')
        # test_list.append(reward_hla) # 收集simulator打分，测试better%
        hla_score = reward_hla

        reward_sum += hla_score
        dynamic_ndcg += env_lla.total_reward
        # if i % 50 == 0:
        #     reward_list.append(reward_sum)

        if upstream_score > 0.8:
            use_upstream += 1

        if hla_score > upstream_score or abs(hla_score-upstream_score) <= 1e-4:
            better += 1

        if PRINT_test:
            print('[upstream vs PP-HRL]: ', upstream_score, hla_score)
            print('[better]', better)
            print('[use upstream]', use_upstream)
            print('--'*10)
            # if i % 50 == 0:
            #     tt = input('pause')

    # f = open('./data/result/criteo_farFill/pp_hddpg_reward.txt', 'w')
    # print(reward_list, file=f)

    print('[avg NDCG@6]: ', dynamic_ndcg / (MAX_EPISODES * (1 - split_rate)))
    print('[avg simulator score]: ', reward_sum/(MAX_EPISODES*(1-split_rate)))
    print('[better percentage]: ', better/(MAX_EPISODES*(1-split_rate)))
    print('------'*5)

    return base_gmv_list, test_gmv_list, better


if __name__ == '__main__':
    base_gmv_list, test_gmv_list, better = train_hDDPG_with_pvpredict()
    # print(base_gmv_list, test_gmv_list, better)


# 500 64 500 64
# [better] 1069
# [use upstream] 272

# 1000 128 1000 128
# [better] 1072
# [use upstream] 277