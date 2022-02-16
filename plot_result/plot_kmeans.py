import matplotlib.pyplot as plt
import numpy as np
import config

N = config.data_size/10

fig, ax = plt.subplots()
font = {'size': 13}
# plt.title('avg_reward_repeat_action')
plt.xlabel("N", font)
plt.ylabel("Average Reward", font)

# f1 = open('./data/reward/avg_reward_kmeans_cossim_size20000_0407.txt','r')
# line = f1.readline().strip('\n').strip('[').strip(']')
# reward1 = line.split(', ')
# for i in range(len(reward1)):
#     reward1[i] = eval(reward1[i])

# plt.plot(np.arange(0,len(reward1),1), reward1, '-', label='avg_reward_kmeans_cossim_size20000', linewidth=1)
#
# f1 = open('./data/reward/avg_reward_kmeans_cossim_size30000_0407.txt','r')
# line = f1.readline().strip('\n').strip('[').strip(']')
# reward1 = line.split(', ')
# for i in range(len(reward1)):
#     reward1[i] = eval(reward1[i])
#
# plt.plot(np.arange(0,len(reward1),1), reward1, '-', label='avg_reward_kmeans_cossim_size30000', linewidth=1)
#
f1 = open('./data/reward/avg_reward_kmeans_cossim_size50000_0407.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward1 = line.split(', ')
for i in range(len(reward1)):
    reward1[i] = eval(reward1[i])

plt.plot(np.arange(0,len(reward1),1), reward1, '-', label='avg_reward_kmeans_cossim_size50000', linewidth=1)

f1 = open('./data/reward/avg_reward_kmeans_incre5000_size50000_0411.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward1 = line.split(', ')
for i in range(len(reward1)):
    reward1[i] = eval(reward1[i])

plt.plot(np.arange(0,len(reward1),1), reward1, '-', label='avg_reward_kmeans_incre5000_size50000_0411', linewidth=1)

f1 = open('./data/reward/avg_reward_kmeans_cossim_size100000_0408.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward1 = line.split(', ')
for i in range(len(reward1)):
    reward1[i] = eval(reward1[i])

plt.plot(np.arange(0,len(reward1),1), reward1, '-', label='avg_reward_kmeans_cossim_size100000', linewidth=1)

f1 = open('./data/reward/avg_reward_kmeans_incre5000_size100000_0411.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward1 = line.split(', ')
for i in range(len(reward1)):
    reward1[i] = eval(reward1[i])

plt.plot(np.arange(0,len(reward1),1), reward1, '-', label='avg_reward_kmeans_incre5000_size100000_0411', linewidth=1)

f1 = open('./data/reward/avg_reward_kmeans_size20000_0411.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward1 = line.split(', ')
for i in range(len(reward1)):
    reward1[i] = eval(reward1[i])

plt.plot(np.arange(0,len(reward1),1), reward1, '-', label='avg_reward_kmeans_size20000_0411', linewidth=1)

f1 = open('./data/reward/avg_reward_kmeans_incre5000_size20000_0411.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward1 = line.split(', ')
for i in range(len(reward1)):
    reward1[i] = eval(reward1[i])

plt.plot(np.arange(0,len(reward1),1), reward1, '-', label='avg_reward_kmeans_incre5000_size20000_0411', linewidth=1)

f1 = open('./data/reward/avg_reward_kmeans_incre5000_size20000_maxnorm_0419.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward1 = line.split(', ')
for i in range(len(reward1)):
    reward1[i] = eval(reward1[i])

plt.plot(np.arange(0,len(reward1),1), reward1, '-', label='avg_reward_kmeans_incre5000_size20000_maxnorm_0419', linewidth=1)

# plt.xticks([0, 200, 400, 600, 800, 1000], ['0', '2000', '4000', '6000', '8000', '10000'])
plt.ylim(3,13)
plt.legend()
plt.show()