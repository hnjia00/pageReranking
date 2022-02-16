import matplotlib.pyplot as plt
import numpy as np
import config

N = config.data_size

fig, ax = plt.subplots()
font = {'size': 13}
# plt.title('avg_reward_repeat_action')
plt.xlabel("N", font)
plt.ylabel("Average Reward", font)

f1 = open('./data/reward/avg_reward_no_fixednet.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward1 = line.split(', ')
for i in range(len(reward1)):
    reward1[i] = eval(reward1[i])

# plt.plot(np.arange(0,len(reward1),1), reward1, '-', label='avg_reward_no_fixednet_0401', linewidth=1)

f1 = open('./data/reward/avg_reward_fixednet.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward1 = line.split(', ')
for i in range(len(reward1)):
    reward1[i] = eval(reward1[i])

# plt.plot(np.arange(0,len(reward1),1), reward1, '-', label='avg_reward_fixednet_0401', linewidth=1)

f1 = open('./data/reward/avg_reward_fixed_0407.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward1 = line.split(', ')
for i in range(len(reward1)):
    reward1[i] = eval(reward1[i])

# plt.plot(np.arange(0,len(reward1),1), reward1, '-', label='avg_reward_fixednet_0407', linewidth=1)
#
f1 = open('./data/reward/avg_reward_cosprop_0407.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward1 = line.split(', ')
for i in range(len(reward1)):
    reward1[i] = eval(reward1[i])

# plt.plot(np.arange(0,N,1), reward1, '-', label='avg_reward_cosprop_0407', linewidth=1)
#
f1 = open('./data/reward/avg_reward_cosprop_0407_1.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward2 = line.split(', ')
for i in range(len(reward2)):
    reward2[i] = eval(reward2[i])

# plt.plot(np.arange(0,N,1), reward1, '-', label='avg_reward_cosprop_0407_1', linewidth=1)
#
f1 = open('./data/reward/avg_reward_cosprop_0407_2.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward3 = line.split(', ')
for i in range(len(reward3)):
    reward3[i] = eval(reward3[i])

sum = []
for i in range(len(reward1)):
    sum.append((reward1[i]+reward3[i])/2)
plt.plot(np.arange(0,len(reward1),1), sum, '-', label='avg_reward_cossim_0407', linewidth=1)

f1 = open('./data/reward/avg_reward_dbscan_cosprop_0407.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward1 = line.split(', ')
for i in range(len(reward1)):
    reward1[i] = eval(reward1[i])

# plt.plot(np.arange(0,N,1), reward1, '-', label='avg_reward_dbscan_cosprop_0407', linewidth=1)

f1 = open('./data/reward/avg_reward_dbscan_cosprop_0407_1.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward2 = line.split(', ')
for i in range(len(reward2)):
    reward2[i] = eval(reward2[i])

# plt.plot(np.arange(0,N,1), reward1, '-', label='avg_reward_dbscan_cosprop_0407_1', linewidth=1)

f1 = open('./data/reward/avg_reward_dbscan_cosprop_0407_2.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward3 = line.split(', ')
for i in range(len(reward3)):
    reward3[i] = eval(reward3[i])

sum = []
for i in range(len(reward1)):
    sum.append((reward1[i]+reward2[i]+reward3[i])/3)

plt.plot(np.arange(0,len(reward1),1), sum, '-', label='avg_reward_dbscan_cossim_0407', linewidth=1)

f1 = open('./data/reward/avg_reward_kmeans_cosprop_0407.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward1 = line.split(', ')
for i in range(len(reward1)):
    reward1[i] = eval(reward1[i])

# plt.plot(np.arange(0,N,1), reward1, '-', label='avg_reward_dbscan_cosprop_0407', linewidth=1)

f1 = open('./data/reward/avg_reward_kmeans_cosprop_0407_1.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward2 = line.split(', ')
for i in range(len(reward2)):
    reward2[i] = eval(reward2[i])

# plt.plot(np.arange(0,N,1), reward1, '-', label='avg_reward_dbscan_cosprop_0407_1', linewidth=1)

f1 = open('./data/reward/avg_reward_kmeans_cosprop_0407_2.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward3 = line.split(', ')
for i in range(len(reward3)):
    reward3[i] = eval(reward3[i])

sum = []
for i in range(len(reward1)):
    sum.append((reward1[i]+reward2[i]+reward3[i])/3)

plt.plot(np.arange(0,len(reward1),1), sum, '-', label='avg_reward_kmeans_cossim_0407', linewidth=1)

f1 = open('./data/reward/avg_reward_L2dis_0519.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward3 = line.split(', ')
for i in range(len(reward3)):
    reward3[i] = eval(reward3[i])

plt.plot(np.arange(0,len(reward3)/10,1), reward3[:500], '-', label='avg_reward_L2dis_0519', linewidth=1)

# plt.xticks([0, 200, 400, 600, 800, 1000], ['0', '2000', '4000', '6000', '8000', '10000'])
plt.ylim(3,12)
plt.legend()
plt.show()