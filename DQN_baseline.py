from open_data.config import *
from open_data.data_process import *
import tensorflow as tf
import numpy as np
import random
from open_data.h_DDPG import meta_controller_DDPG, controller_DDPG
# from h_DDPG import dqn_DDPG, controller_DDPG
from open_data.Env_model import *
from sklearn.externals import joblib
from open_data.simulator.load_simulator import load_simulator
import torch
from tqdm import tqdm, trange

from open_data.DQN_Agent import lower_DQN

np.random.seed(1)
tf.set_random_seed(1)

sess = tf.Session()

# action_classifier = joblib.load('./data/action_classifier/dbscan/dbscan_action_classfier_eps0.1.m')
action_classifier = joblib.load('./data/action_classifier/kmeans/kmeans_hierachical_action_classfier.m')

# 16->6 recommendation场景建模
env_rec = Env()
# 6->2*3 rerank场景建模
env_rer = Env_rerank()

# rec 16->6
dqn = lower_DQN(16, state_dim)
# rerank 6
ddpg =  controller_DDPG(action_dim, c_state_dim, action_bound)

# pv-wise simulator
simulator = load_simulator()

def train_hDDPG_with_simulator(action_mode='cossim', PRINT = False):
    global EPSILON
    global c_EPSILON

    sess.run(tf.global_variables_initializer())

    action_cnt = 0
    action_feature_collecter = []
    Reward_meta = []
    meta_ep_reward = 0
    Reward_controller = []
    controller_ep_reward = 0
    ndcg_ddpg = []
    ndcg_ep_reward = 0

    ddpg_deal = 0
    hddpg_deal = 0
    hddpg_outperform_cnt = 0
    ddpg_outperform_cnt = 0
    hddpg_equal_cnt = 0
    ddpg_equal_cnt = 0
    use_upstream = 0

    # train
    for i in trange(0, int(split_rate * MAX_EPISODES)):
        goal = []
        goal_index = []

        s, l = env_rec.reset()
        s0 = s

        # dqn 16->6
        mask = np.ones((16,))
        for j in range(env_rec.k):
            # 因为涉及到随机选，所以要保证动作不会重复选择
            while True:
                a_ = dqn.choose_action(s, mask)

                if random.random() < EPSILON:
                    a_ = random.randint(0, canditate_size-1)
                    # 随机选取动作时使用cluster类中心替代a
                    # a = action_classifier.cluster_centers_[a_]
                if a_ not in goal_index:
                    goal_index.append(a_)
                    mask[a_] = 0
                    # TODO: 应该match到16个候选中的doc
                    a = s[action_dim * a_: action_dim * a_ + action_dim]
                    goal.append(a)
                    break

            s_, r, done = env_rec.step_in_one_line(a_)

            dqn.store_transition(s, a_, r / 10, s_)

            if dqn.memory_counter > MEMORY_CAPACITY:
                # print("action: ", a_)
                if a_ < env_rec.k: action_cnt += 1
                if EPSILON > 0.05: EPSILON *= .9995  # decay the action randomness
                dqn.learn()

            meta_ep_reward += r
            s = s_

            if done:
                break

        # ddpg_controler 6-> rerank 6
        rerank_set = []
        c_s = env_rer.reset(state=s0, goal=goal, pointer=i)
        for k in range(env_rer.k):
            while True:
                c_a_proto = ddpg.choose_action(c_s)
                # TODO: match_action
                if action_mode == 'L2':
                    c_a_index = env_rer.match_action_L2(c_a_proto) # L2 match
                elif action_mode == 'cossim':
                    c_a_index = env_rer.match_action_cos(c_a_proto)
                else:
                    c_a_index = env_rer.match_action_cluster(c_a_proto, action_classifier)
                if random.random() < c_EPSILON:
                    c_a_index = random.randint(0, env_rer.k - 1)

                c_a = goal[c_a_index]

                if c_a_index not in rerank_set:
                    rerank_set.append(c_a_index)
                    break

            c_s_, c_r, c_done = env_rer.step(action=k, action_feature=c_a, pointer=i)

            ddpg.store_transition(c_s, c_a, c_r*100, c_s_)

            if ddpg.pointer > c_MEMORY_CAPACITY:
                # print("action: ", c_a_index)
                if c_EPSILON > 0.05: c_EPSILON *= .9995  # decay the action randomness
                ddpg.learn()

            c_s = c_s_

            if c_done != False:
                controller_ep_reward += c_done[0]
                # print('Episode:', i, 'Avg_Reward_controller: %.4f' % (controller_ep_reward/(i+1)), 'Explore: %.2f' % c_EPSILON)
                ndcg_ep_reward += c_done[1]
                # print('Episode:', i, 'Avg_Reward_ddpg_baseline: %.4f' % (ndcg_ep_reward / (i + 1)))
                break

    # saver = tf.train.Saver()
    # saver.save(sess, "./data/model/HDDPG_DDPG_with_kmeans_0615/HDDPG_and_DDPG_0615.ckpt")

    # test
    reward_ddpg_list = []
    reward_ddpg = 0
    reward_hddpg_list = []
    reward_hddpg = 0
    ndcg_ddpg = 0
    ndcg_hddpg = 0
    for i in trange(int(split_rate * MAX_EPISODES), MAX_EPISODES):
        goal = []
        goal_index = []

        s, l = env_rec.reset()
        s0 = s
        # dqn 16->6
        mask = np.ones((16, ))
        for j in range(env_rec.k):
            # 因为涉及到随机选，所以要保证动作不会重复选择
            while True:
                a = dqn.choose_action(s, mask)

                if random.random() < 0.1:
                    a_ = random.randint(0, canditate_size - 1)

                if a_ not in goal_index:
                    goal_index.append(a_)
                    mask[a_] = 0
                    # TODO: 应该match到16个候选中的doc
                    a = s[action_dim * a_: action_dim * a_ + action_dim]
                    goal.append(a)
                    break

            s_, r, done = env_rec.step_in_one_line(a_)

            meta_ep_reward += r
            s = s_

            if done:
                # simulator -> log pv-wise prob
                log_pv_vec = s0[:env_rec.k*item_dim].astype('float64')
                log_pv_vec = np.reshape(log_pv_vec, (1, 28*6))
                outputs = simulator(torch.Tensor(log_pv_vec)).detach().numpy()[0]
                log_pv_prob = np.exp(outputs[1]) / (np.exp(outputs[0]) + np.exp(outputs[1]))

                # simulator -> DDPG pv-wise prob
                pv_vec = np.reshape(goal,(1, 28*6)).astype(np.float64)
                outputs = simulator(torch.Tensor(pv_vec)).detach().numpy()[0]
                pv_wise_ddpg_prob = np.exp(outputs[1]) / (np.exp(outputs[0]) + np.exp(outputs[1]))
                break

        # ddpg_controler 6-> rerank 6
        rerank_set = []
        c_s = env_rer.reset(state=s0, goal=goal, pointer=i)
        for k in range(env_rer.k):
            while True:
                c_a_proto = ddpg.choose_action(c_s)

                if action_mode == 'L2':
                    c_a_index = env_rer.match_action_L2(c_a_proto)  # L2 match
                elif action_mode == 'cossim':
                    c_a_index = env_rer.match_action_cos(c_a_proto)
                else:
                    c_a_index = env_rer.match_action_cluster(c_a_proto, action_classifier)

                if random.random() < 0.1:
                    c_a_index = random.randint(0, env_rer.k - 1)

                c_a = goal[c_a_index]

                if c_a_index not in rerank_set:
                    rerank_set.append(c_a_index)
                    break

            c_s_, c_r, c_done = env_rer.step(action=k, action_feature=c_a, pointer=i)

            c_s = c_s_

            if c_done != False:
                # simulator -> H-DDPG pv-wise mark
                pv_vec = c_s[28*6:].astype(np.float64)
                pv_vec = np.reshape(pv_vec, (1, 28 * 6))
                outputs = simulator(torch.Tensor(pv_vec)).detach().numpy()[0]
                pv_wise_hddpg_prob = np.exp(outputs[1]) / (np.exp(outputs[0]) + np.exp(outputs[1]))
                # print('pv_wise_hddpg_prob', float(pv_wise_hddpg_prob))
                # price_sum = 0
                # for goal_i in goal:
                #     price_sum += eval(goal_i[-5])
                # hddpg_deal += pv_wise_hddpg_prob * price_sum / env_rer.k
                break

        reward_ddpg += pv_wise_ddpg_prob
        reward_hddpg += pv_wise_hddpg_prob

        ndcg_ddpg += c_done[1]
        ndcg_hddpg += c_done[0]

        if i % 50 == 0:
            reward_ddpg_list.append(reward_ddpg)
            reward_hddpg_list.append(reward_hddpg)


        if pv_wise_hddpg_prob >= log_pv_prob:
            hddpg_outperform_cnt += 1
        if pv_wise_ddpg_prob >= log_pv_prob:
            ddpg_outperform_cnt += 1

    # f = open('./data/result/criteo_farFill/ddpg_reward.txt', 'w')
    # print(reward_ddpg_list, file=f)
    # f = open('./data/result/criteo_farFill/ts_hddpg_reward.txt', 'w')
    # print(reward_hddpg_list, file=f)
    print("HDDPG")
    print('[avg NDCG@6]: ', ndcg_hddpg / (MAX_EPISODES * (1 - split_rate)))
    print('[avg simulator score]: ', reward_hddpg/(MAX_EPISODES*(1-split_rate)))
    print('[better percentage]: ', hddpg_outperform_cnt/(MAX_EPISODES*(1-split_rate)))
    print('------'*5)

    print("DQN")
    print('[avg NDCG@6]: ', ndcg_ddpg / (MAX_EPISODES * (1 - split_rate)))
    print('[avg simulator score]: ', reward_ddpg/(MAX_EPISODES*(1-split_rate)))
    print('[better percentage]: ', ddpg_outperform_cnt/(MAX_EPISODES*(1-split_rate)))
    print('------'*5)


if __name__ == '__main__':
    for _ in range(1):
        train_hDDPG_with_simulator(action_mode='cossim', PRINT=False)

# 1000 128 1000 128
# [h-ddpg outperform]:  823
# [h-ddpg equal]:  0
# [ddpg outperform]:  107
# [ddpg equal]:  0

# far data
# [use upstream] 369
# [h-ddpg outperform]:  1127
# [h-ddpg equal]:  1
# [ddpg outperform]:  1162
# [ddpg equal]:  2
