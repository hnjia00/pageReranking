from open_data.config import *
from open_data.data_process import *
import tensorflow as tf
import numpy as np
import random
from open_data.HDDPG_Agent import meta_controller_DDPG, controller_DDPG
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

# 16->6 recommendation场景建模
# env_rec = Env()
# 6->2*3 rerank场景建模
# env_rer = Env_rerank()
# HLA pv预测场景建模
env_hla = Env_high_level()
env_lla = Env_low_level()

# pv-wise simulator
simulator = load_simulator()

# rec 16->6
# ddpg_meta_controller = meta_controller_DDPG(action_dim, state_dim, action_bound)
# rerank 6
# ddpg_controller = controller_DDPG(c_action_dim, c_state_dim, c_action_bound)

def train_hDDPG_with_simulator(action_mode='kmeans', PRINT = False):
    global EPSILON
    global c_EPSILON

    # 16->6 recommendation场景建模
    env_rec = Env()
    # 6->2*3 rerank场景建模
    env_rer = Env_rerank()

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

    # train
    for i in trange(0, int(split_rate * MAX_EPISODES)):
        goal = []
        goal_index = []

        s, l = env_rec.reset()
        s0 = s

        # ddpg_meta_controller 16->6
        for j in range(env_rec.k):
            # 因为涉及到随机选，所以要保证动作不会重复选择
            while True:
                a = ddpg_meta_controller.choose_action(s)
                if action_mode == 'L2':
                    a_ = env_rec.match_action_L2(a, s) # L2 match
                elif action_mode == 'cossim':
                    a_ = env_rec.match_action_cos(a, s)
                else:
                    a_ = env_rec.match_action_cluster(a, s, action_classifier)
                if random.random() < EPSILON:
                    a_ = random.randint(0, canditate_size-1)
                    # 随机选取动作时使用cluster类中心替代a
                    # a = action_classifier.cluster_centers_[a_]
                if a_ not in goal_index:
                    goal_index.append(a_)
                    # TODO: 应该match到16个候选中的doc
                    a = s[action_dim * a_: action_dim * a_ + action_dim]
                    goal.append(a)
                    break

            s_, r, done = env_rec.step_in_one_line(a_)

            ddpg_meta_controller.store_transition(s, a, r / 10, s_)

            if ddpg_meta_controller.pointer > MEMORY_CAPACITY:
                # print("action: ", a_)
                if a_ < env_rec.k: action_cnt += 1
                if EPSILON > 0.05: EPSILON *= .9995  # decay the action randomness
                ddpg_meta_controller.learn()

                # incremental update action classifier
                if i > 5000 and ddpg_meta_controller.pointer % MEMORY_CAPACITY == 0 and INCRE_update:
                    action_feature = extract_action_feature_from_memory(ddpg_meta_controller.memory)
                    action_feature_collecter.extend(action_feature)
                    if len(action_feature_collecter) % 5000 == 0:
                        action_classifier.fit(action_feature)
                        joblib.dump(action_classifier, './data/action_classifier/kmeans/kmeans_hierachical_action_classfier' + str(
                            ddpg_meta_controller.pointer) + '.m')
                        # action_feature_collecter = []
            meta_ep_reward += r
            s = s_

            if done:
                # print('Episode:', i, ' Reward_meta: %i' % int(meta_ep_reward), 'Explore: %.2f' % EPSILON,
                #       action_cnt / (6 * (i + 1) - MEMORY_CAPACITY))
                if i % 10 == 0:
                    Reward_meta.append(meta_ep_reward / (i + 1))
                    # Reward_meta.append(action_cnt/(6*(i+1)-MEMORY_CAPACITY))
                break

        # ddpg_controler 6-> rerank 6
        rerank_set = []
        c_s = env_rer.reset(state=s0, goal=goal, pointer=i)
        for k in range(env_rer.k):
            while True:
                c_a_proto = ddpg_controller.choose_action(c_s)
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

            ddpg_controller.store_transition(c_s, c_a, c_r*100, c_s_)

            if ddpg_controller.pointer > c_MEMORY_CAPACITY:
                # print("action: ", c_a_index)
                if c_EPSILON > 0.05: c_EPSILON *= .9995  # decay the action randomness
                ddpg_controller.learn()

            c_s = c_s_

            if c_done != False:
                controller_ep_reward += c_done[0]
                # print('Episode:', i, 'Avg_Reward_controller: %.4f' % (controller_ep_reward/(i+1)), 'Explore: %.2f' % c_EPSILON)
                ndcg_ep_reward += c_done[1]
                # print('Episode:', i, 'Avg_Reward_ddpg_baseline: %.4f' % (ndcg_ep_reward / (i + 1)))

                if PRINT and i%500 == 0:
                    f = open('./data/reward/avg_ndcg_hddpg_static_ndcg_0519.txt', 'w')
                    print(Reward_controller, file=f)
                    f = open('./data/reward/avg_ndcg_ddpg_static_ndcg_0519.txt', 'w')
                    print(ndcg_ddpg, file=f)

                if i % 10 == 0:
                    Reward_controller.append(controller_ep_reward / (i + 1))
                    ndcg_ddpg.append(ndcg_ep_reward / (i+1)) # 收集单层ddpg的ndcg
                break

    # saver = tf.train.Saver()
    # saver.save(sess, "./data/model/HDDPG_DDPG_with_kmeans_0615/HDDPG_and_DDPG_0615.ckpt")

    # test
    for i in trange(int(split_rate * MAX_EPISODES), MAX_EPISODES):
        goal = []
        goal_index = []

        s, l = env_rec.reset()
        s0 = s
        # ddpg_meta_controller 16->6
        for j in range(env_rec.k):
            # 因为涉及到随机选，所以要保证动作不会重复选择
            while True:
                a = ddpg_meta_controller.choose_action(s)
                if action_mode == 'L2':
                    a_ = env_rec.match_action_L2(a, s) # L2 match
                else:
                    a_ = env_rec.match_action_cluster(a, s, action_classifier)

                if a_ not in goal_index:
                    goal_index.append(a_)
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
                log_pv_prob = simulator(torch.Tensor(log_pv_vec))

                # simulator -> DDPG pv-wise prob
                pv_vec = np.reshape(goal,(28*6)).astype(np.float64)
                pv_wise_ddpg_prob = simulator(torch.Tensor(pv_vec))
                # print('pv_wise_ddpg_prob', float(pv_wise_ddpg_prob))
                # sum E[turnover]
                price_sum = 0
                for goal_i in goal:
                    price_sum += eval(goal_i[-5])
                ddpg_deal += pv_wise_ddpg_prob * price_sum / env_rec.k
                break

        # ddpg_controler 6-> rerank 6
        rerank_set = []
        c_s = env_rer.reset(state=s0, goal=goal, pointer=i)
        for k in range(env_rer.k):
            while True:
                c_a_proto = ddpg_controller.choose_action(c_s)

                if action_mode == 'L2':
                    c_a_index = env_rer.match_action_L2(c_a_proto)  # L2 match
                else:
                    c_a_index = env_rer.match_action_cluster(c_a_proto, action_classifier)

                if random.random() < c_EPSILON:
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
                pv_wise_hddpg_prob = simulator(torch.Tensor(pv_vec))
                # print('pv_wise_hddpg_prob', float(pv_wise_hddpg_prob))
                price_sum = 0
                for goal_i in goal:
                    price_sum += eval(goal_i[-5])
                hddpg_deal += pv_wise_hddpg_prob * price_sum / env_rer.k
                break


        if pv_wise_hddpg_prob > log_pv_prob:
            hddpg_outperform_cnt += 1
        elif pv_wise_ddpg_prob > log_pv_prob:
            ddpg_outperform_cnt += 1
        elif pv_wise_hddpg_prob == log_pv_prob:
            hddpg_equal_cnt += 1
        elif pv_wise_ddpg_prob == log_pv_prob:
            ddpg_equal_cnt += 1


    print('[h-ddpg outperform]: ', hddpg_outperform_cnt)
    print('[h-ddpg equal]: ', hddpg_equal_cnt)
    print('[ddpg outperform]: ', ddpg_outperform_cnt)
    print('[ddpg equal]: ', ddpg_equal_cnt)

    # print('[h-ddpg outperform ddpg]: ', outperform_cnt)
    # print('[h-ddpg == ddpg]: ', equal_cnt)
    print('[ddpg deal]: ', float(ddpg_deal.data))
    print('[h-ddpg deal]: ', float(hddpg_deal.data))
    print('-----'*10)

def train_hDDPG(env_rec, env_rer, PRINT = True):
    global EPSILON
    global c_EPSILON
    action_cnt = 0
    action_feature_collecter = []

    # rec 16->6
    ddpg_meta_controller = meta_controller_DDPG(action_dim, state_dim, action_bound)
    # rerank 6
    ddpg_controller = controller_DDPG(c_action_dim, c_state_dim, c_action_bound)

    sess.run(tf.global_variables_initializer())

    Reward_meta = []
    meta_ep_reward = 0
    Reward_controller = []
    controller_ep_reward = 0
    ndcg_ddpg = []
    ndcg_ep_reward = 0
    for i in range(MAX_EPISODES):
        goal = []
        goal_index = []

        s, l = env_rec.reset()
        s0 = s

        # ddpg_meta_controller 16->6
        for j in range(env_rec.k):
            # 因为涉及到随机选，所以要保证动作不会重复选择
            while True:
                a = ddpg_meta_controller.choose_action(s)
                a_ = env_rec.match_action_cluster(a, s, action_classifier)
                if random.random() < EPSILON:
                    a_ = random.randint(0, canditate_size-1)
                    # 随机选取动作时使用cluster类中心替代a
                    a = action_classifier.cluster_centers_[a_]
                if a_ not in goal_index:
                    goal_index.append(a_)
                    # TODO: 应该match到16个候选中的doc
                    a = s[action_dim * a_: action_dim * a_ + action_dim]
                    goal.append(a)
                    break

            s_, r, done = env_rec.step_in_one_line(a_)

            ddpg_meta_controller.store_transition(s, a, r / 10, s_)

            if ddpg_meta_controller.pointer > MEMORY_CAPACITY:
                # print("action: ", a_)
                if a_ < env_rec.k: action_cnt += 1
                if EPSILON > 0.05: EPSILON *= .9995  # decay the action randomness
                ddpg_meta_controller.learn()

                # incremental update action classifier
                if i > 5000 and ddpg_meta_controller.pointer % MEMORY_CAPACITY == 0 and INCRE_update:
                    action_feature = extract_action_feature_from_memory(ddpg_meta_controller.memory)
                    action_feature_collecter.extend(action_feature)
                    if len(action_feature_collecter) % 5000 == 0:
                        action_classifier.fit(action_feature)
                        joblib.dump(action_classifier, './data/action_classifier/kmeans/kmeans_hierachical_action_classfier' + str(
                            ddpg_meta_controller.pointer) + '.m')
                        # action_feature_collecter = []
            meta_ep_reward += r
            s = s_

            if done:
                # print('Episode:', i, ' Reward_meta: %i' % int(meta_ep_reward), 'Explore: %.2f' % EPSILON,
                #       action_cnt / (6 * (i + 1) - MEMORY_CAPACITY))
                if i % 10 == 0:
                    Reward_meta.append(meta_ep_reward / (i + 1))
                    # Reward_meta.append(action_cnt/(6*(i+1)-MEMORY_CAPACITY))
                break

        # ddpg_controler 6-> rerank 6
        rerank_set = []
        c_s = env_rer.reset(state=s0, goal=goal, pointer=i)
        for k in range(env_rer.k):
            while True:
                c_a_proto = ddpg_controller.choose_action(c_s)
                # TODO: match_action
                c_a_index = env_rer.match_action_cluster(c_a_proto, action_classifier)
                if random.random() < c_EPSILON:
                    c_a_index = random.randint(0, env_rer.k - 1)

                c_a = goal[c_a_index]

                if c_a_index not in rerank_set:
                    rerank_set.append(c_a_index)
                    break

            c_s_, c_r, c_done = env_rer.step(action=k, action_feature=c_a, pointer=i)

            ddpg_controller.store_transition(c_s, c_a, c_r*100, c_s_)

            if ddpg_controller.pointer > c_MEMORY_CAPACITY:
                # print("action: ", c_a_index)
                if c_EPSILON > 0.05: c_EPSILON *= .9995  # decay the action randomness
                ddpg_controller.learn()

            c_s = c_s_

            if c_done != False:
                controller_ep_reward += c_done[0]
                print('Episode:', i, 'Avg_Reward_controller: %.4f' % (controller_ep_reward/(i+1)), 'Explore: %.2f' % c_EPSILON)
                ndcg_ep_reward += c_done[1]
                print('Episode:', i, 'Avg_Reward_ddpg_baseline: %.4f' % (ndcg_ep_reward / (i + 1)))

                if PRINT and i%500 == 0:
                    f = open('./data/reward/avg_ndcg_hddpg_static_ndcg_0519.txt', 'w')
                    print(Reward_controller, file=f)
                    f = open('./data/reward/avg_ndcg_ddpg_static_ndcg_0519.txt', 'w')
                    print(ndcg_ddpg, file=f)

                if i % 10 == 0:
                    Reward_controller.append(controller_ep_reward / (i + 1))
                    ndcg_ddpg.append(ndcg_ep_reward / (i+1)) # 收集单层ddpg的ndcg
                break

def train_hDDPG_with_pvpredict(PRINT_train = False, PRINT_test = True):
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
        # low_level_agent.reset()
        for j in range(env_lla.k):
            while True:
                proto_item_feature = low_level_agent.choose_action(state_lla)
                proto_item_feature_index = env_lla.match_action_cluster(proto_item_feature, action_classifier)
                if random.random() < c_EPSILON:
                    proto_item_feature_index = random.randint(0, env_lla.canditate_size-1)
                    # 随机选取动作时使用cluster类中心替代a
                    proto_item_feature = action_classifier.cluster_centers_[proto_item_feature_index]
                if proto_item_feature_index not in env_lla.select_index:
                    break

            state_lla_, reward_lla, done = env_lla.step(action_index=proto_item_feature_index, pointer=i)

            low_level_agent.store_transition(state_lla, proto_item_feature, reward_lla, state_lla_)

            if low_level_agent.pointer > c_MEMORY_CAPACITY:
                low_level_agent.learn()
                # incremental update action classifier
                if i > 5000 and low_level_agent.pointer % c_MEMORY_CAPACITY == 0 and INCRE_update:
                    action_feature = extract_action_feature_from_lla_memory(low_level_agent.memory)
                    action_feature_lla.extend(action_feature)
                    if len(action_feature_lla) % 5000 == 0:
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

        # high_level_agent.store_transition(s=state_hla, a=action_feature_lla, r=reward_hla, s_=state_hla_) # 存放虚假的pv
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
    test_list = []
    for i in trange(int(split_rate * MAX_EPISODES), MAX_EPISODES):
        print('i',i)

        state_hla, label_canditate_16 = env_hla.reset()

        proto_pv = high_level_agent.choose_action(state_hla)

        # low level environment 16->6
        lla_reward_list = []
        state_lla = env_lla.reset(state=state_hla, label=label_canditate_16, goal=proto_pv, pointer=i)
        for j in range(env_lla.k):
            while True:
                proto_item_feature = low_level_agent.choose_action(state_lla)
                proto_item_feature_index = env_lla.match_action_cluster(proto_item_feature, action_classifier)
                if random.random() < 0.05:
                    proto_item_feature_index = random.randint(0, env_lla.canditate_size-1)
                    # 随机选取动作时使用cluster类中心替代a
                    proto_item_feature = action_classifier.cluster_centers_[proto_item_feature_index]
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

        if i == MAX_EPISODES-1: break
        state_hla_, label_, reward_hla, done = env_hla.step(state=state_hla, goal=proto_pv, action=env_lla.select_vec,
                                                            lla_reward_list=lla_reward_list, mode="test")
        test_list.append(reward_hla) # 收集simulator打分，测试better%

        if PRINT_test:
            sum_hla_reward_list.append(sum_hla_reward_list[-1]+reward_hla)
            print('[HLA average reward]:', sum_hla_reward_list[-1])

    return test_list

def test_upstream_baseline(PRINT = True):
    env_baseline = Env_upstream()

    reward_list = [0]
    for i in range(MAX_EPISODES):
        if i < MAX_EPISODES * split_rate: continue

        state, label = env_baseline.reset()
        reward = env_baseline.to_reward(state, reward_mode='simulator_score')
        # avg_reward = (reward[-1]*i + reward) / (i + 1)
        # if PRINT:
        #     print('[Upstream pv]', avg_reward)
        # reward_list.append(float(avg_reward))

        # 去除purchase=1
        # flag = False
        # for j in range(6):
        #     if label[j][2] == 1:
        #         flag = True
        #
        # if flag:
        #     reward = -1
        reward_list.append(reward)

    if PRINT:
        # f = open('./data/reward/avg_upstream_pv_reward_0603.txt', 'w')
        # print(reward_list, file=f)
        print(sum(reward_list))

    return reward_list


if __name__ == '__main__':
    # for _ in range(1):
    #     train_hDDPG_with_simulator(action_mode='cossim', PRINT=False)
    h = train_hDDPG_with_pvpredict()
    b = test_upstream_baseline()
    #
    # f = open('./data/reward/simulator_score_v1_3w_0705.txt', 'w')
    # print(h, file=f)
    # print(b, file=f)

    # f = open('./data/reward/simulator_score_30000_0704.txt', 'r')
    # lines = f.readlines()
    # h = np.array(lines[0].strip('\n').strip('[').strip(']').split(', ')).astype('float64')
    # b = np.array(lines[1].strip('\n').strip('[').strip(']').split(', ')).astype('float64')

    better = 0
    equal = 0
    p = 0
    use_upstream = 0
    # hddpg_cnt = {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0, 1: 0}
    # upstream_cnt = {0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0, 1: 0}
    for i,j in zip(h, b):
        # if j == -1:
        #      p += 1
        if j > 0.8:
            # print(i, j)
            use_upstream += 1
        elif i > j:
            print(i, j)
            better += 1
        elif i == j:
            equal += 1

    #
    # # print(p)
    print('p, use_upstream', p, use_upstream)
    print('better, equal', better, equal)
    # print('hddpg cnt', hddpg_cnt)
    # print('upstream_cnt', upstream_cnt)

# hddpg cnt {0.5: 3333, 0.6: 235, 0.7: 213, 0.8: 130, 0.9: 59, 1: 29}
# upstream_cnt {0.5: 2857, 0.6: 315, 0.7: 299, 0.8: 280, 0.9: 172, 1: 76}
