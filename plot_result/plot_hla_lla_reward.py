import matplotlib.pyplot as plt
import numpy as np
import config

N = config.data_size

fig, ax = plt.subplots()
font = {'size': 13}
# plt.title('avg_reward_repeat_action')
plt.xlabel("N", font)
plt.ylabel("Average Reward", font)

f1 = open('../data/reward/avg_lla_reward_0602.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward1 = line.split(', ')
for i in range(len(reward1)):
    reward1[i] = eval(reward1[i])

plt.plot(np.arange(0,len(reward1)), reward1, '-', label='avg_lla_reward_0602', linewidth=1)

f1 = open('../data/reward/avg_hla_reward_0602.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward1 = line.split(', ')
for i in range(len(reward1)):
    reward1[i] = eval(reward1[i])

# plt.plot(np.arange(0,len(reward1)), reward1, '-', label='avg_hla_reward_0602', linewidth=1)

# plt.xticks([0, 200, 400, 600, 800, 1000], ['0', '2000', '4000', '6000', '8000', '10000'])
# plt.ylim(3,12)
plt.legend()
plt.show()