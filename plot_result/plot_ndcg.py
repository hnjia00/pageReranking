import matplotlib.pyplot as plt
import numpy as np
import config


fig, ax = plt.subplots()
font = {'size': 13}
# plt.title('avg_reward_repeat_action')
plt.xlabel("N", font)
plt.ylabel("Average Ndcg", font)

f1 = open('./data/reward/avg_ndcg_ddpg_static_ndcg_0519.txt','r')
line = f1.readline().strip('\n').strip('[').strip(']')
reward1 = line.split(', ')
for i in range(len(reward1)):
    reward1[i] = eval(reward1[i])
reward1[0] = 0
# plt.plot(np.arange(0,len(reward1),1), reward1, '-', label='ddpg_static_ndcg', linewidth=1)

f2 = open('./data/reward/avg_ndcg_hddpg_static_ndcg_0519.txt','r')
line = f2.readline().strip('\n').strip('[').strip(']')
reward2 = line.split(', ')
for i in range(len(reward2)):
    reward2[i] = eval(reward2[i])
reward2[0] = 0
# plt.plot(np.arange(0,len(reward2),1), reward2, '-', label='hddpg_static_ndcg', linewidth=1)

# plt.plot(np.arange(0,len(reward2),1), np.array(reward2)-np.array(reward1), '-', label='hddpg $-$ ddpg', linewidth=1)
# plt.plot(np.arange(0,len(reward2),1), [0]*len(reward2),'-')

# plt.xticks([0, 200, 400, 600, 800, 1000], ['0', '2000', '4000', '6000', '8000', '10000'])

# print(np.array(reward2)-np.array(reward1))
# plt.ylim(-0.001,0.003)
# plt.legend()
# plt.show()


tick = -1
print(reward1[tick], reward2[tick])
