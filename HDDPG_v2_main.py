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
import torch
from tqdm import tqdm, trange

np.random.seed(1)
tf.set_random_seed(1)

sess = tf.Session()

# action_classifier = joblib.load('./data/action_classifier/dbscan/dbscan_action_classfier_eps0.1.m')
action_classifier = joblib.load('./data/action_classifier/kmeans/kmeans_hierachical_action_classfier.m')

# HLA pv预测场景建模
env_hla = Env_high_level()
env_lla = Env_low_level()

# pv-wise simulator
simulator = load_simulator()

def train_hDDPG_with_pvpredict(PRINT_train = False, PRINT_test = False):
    global EPSILON
    global c_EPSILON

    # pv predict
    high_level_agent = meta_controller_DDPG(action_dim, state_dim, action_bound)
    # 16 -> 6
    low_level_agent = controller_DDPG(c_action_dim, c_state_dim, c_action_bound)

    sess.run(tf.global_variables_initializer())

    sum_lla_reward_list = [0]
    sum_hla_reward_list = [0]
    action_feature_lla = []

    ########################## Train ##########################
    # high level environment proto-pv predict
    for i in trange(int(MAX_EPISODES*split_rate)):
        state_hla, label_canditate_16 = env_hla.reset()

        proto_pv = high_level_agent.choose_action(state_hla)

        # low level environment 16->6
        lla_reward_list = []
        state_lla = env_lla.reset(state=state_hla, label=label_canditate_16, goal=proto_pv, pointer=i)
        low_level_agent.reset()
        for j in range(env_lla.k):
            while True:
                proto_item_feature = low_level_agent.choose_action(state_lla)
                proto_item_feature_index = env_lla.match_action_cluster(proto_item_feature, action_classifier)
                if random.random() < c_EPSILON:
                    c_EPSILON *= 0.995
                    proto_item_feature_index = random.randint(0, env_lla.canditate_size-1)
                    # 随机选取动作时使用cluster类中心替代a
                    proto_item_feature = action_classifier.cluster_centers_[proto_item_feature_index]
                if proto_item_feature_index not in env_lla.select_index:
                    break

            state_lla_, reward_lla, done = env_lla.step(action_index=proto_item_feature_index, pointer=i)
            proto_item_feature = np.squeeze(proto_item_feature)
            low_level_agent.store_transition(state_lla, proto_item_feature, reward_lla, state_lla_)

            if low_level_agent.pointer > c_MEMORY_CAPACITY:
                low_level_agent.learn()
                # incremental update action classifier
                if i > 1000 and low_level_agent.pointer % c_MEMORY_CAPACITY == 0 and INCRE_update:
                    action_feature = extract_action_feature_from_lla_memory(low_level_agent.memory)
                    action_feature_lla.extend(action_feature)
                    if len(action_feature_lla) % 3000 == 0:
                        action_classifier.fit(action_feature)
                        # joblib.dump(action_classifier,
                        #             './data/action_classifier/kmeans/kmeans_hierachical_action_classfier' + str(
                        #                 low_level_agent.pointer) + '.m')

            lla_reward_list.append(reward_lla)  # 记录本轮lla奖励
            if done:
                sum_lla_reward = (env_lla.total_reward + sum_lla_reward_list[-1])*i / (i+1) # 使用total_reward记录lla排好后的总奖励，reward_lla为差分奖励
                sum_lla_reward_list.append(sum_lla_reward) # 记录平均lla奖励

                # print('[LLA average reward]:', sum_lla_reward)
                break

            state_lla = state_lla_

        state_hla_, label_, reward_hla, done = env_hla.step(state=state_hla, goal=proto_pv, action=env_lla.select_vec, lla_reward_list=lla_reward_list)

        # high_level_agent.store_transition(s=state_hla, a=proto_pv, r=reward_hla, s_=state_hla_) # 存放虚假的pv
        high_level_agent.store_transition(s=state_hla, a=env_lla.select_vec, r=reward_hla, s_=state_hla_)

        if high_level_agent.pointer > MEMORY_CAPACITY:
            high_level_agent.learn()

        sum_hla_reward = (sum_hla_reward_list[-1]*i + reward_hla)/(i+1)
        sum_hla_reward_list.append(float(sum_hla_reward))
        # print('[HLA average reward]:', sum_hla_reward)

    if PRINT_train:
        f = open('./data/reward/avg_hla_reward_0602.txt', 'w')
        print(sum_hla_reward_list, file=f)
        f = open('./data/reward/avg_lla_reward_0602.txt', 'w')
        print(sum_lla_reward_list, file=f)

        saver = tf.train.Saver()
        saver.save(sess, "./data/model/HLA_LLA_with_kmeans_0618/HRL_0618.ckpt")

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

        # sum_price = 0
        # for up_i in range(env_hla.k):
        #     sum_price += upstream_pv[up_i*item_dim: (up_i+1)*item_dim][-5]
        # upstream_gmv = sum_price * upstream_score / env_hla.k
        # base_gmv_list += upstream_gmv

        proto_pv = high_level_agent.choose_action(state_hla)

        # low level environment 16->6
        lla_reward_list = []
        state_lla = env_lla.reset(state=state_hla, label=label_canditate_16, goal=proto_pv, pointer=i)
        for j in range(env_lla.k):
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
        state_hla_, label_, reward_hla, done = env_hla.step(state=state_hla, goal=proto_pv, action=env_lla.select_vec, lla_reward_list=lla_reward_list, mode='test')
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