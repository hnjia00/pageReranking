import matplotlib.pyplot as plt
import numpy as np

f = open('./criteo_normalFill/ddpg_reward.txt', 'r')
ddpg_reward = f.readline().strip('\n').strip('[').strip(']').split(', ')
ddpg_reward = np.array(ddpg_reward).astype(np.float64)
for i in range(len(ddpg_reward)):
    ddpg_reward[i] = ddpg_reward[i]/((i+1)*50)
print(ddpg_reward[-1])
plt.plot(np.arange(0, len(ddpg_reward)), ddpg_reward, label='ddpg')

f = open('./criteo_normalFill/pp_hddpg_reward.txt', 'r')
pp_hddpg_reward = f.readline().strip('\n').strip('[').strip(']').split(', ')
pp_hddpg_reward = np.array(pp_hddpg_reward).astype(np.float64)
for i in range(len(pp_hddpg_reward)):
    pp_hddpg_reward[i] = pp_hddpg_reward[i]/((i+1)*50)
print(pp_hddpg_reward[-1])
plt.plot(np.arange(0, len(pp_hddpg_reward)), pp_hddpg_reward, label='pp_hddpg')

f = open('./criteo_normalFill/ts_hddpg_reward.txt', 'r')
ts_hddpg_reward = f.readline().strip('\n').strip('[').strip(']').split(', ')
ts_hddpg_reward = np.array(ts_hddpg_reward).astype(np.float64)
for i in range(len(ddpg_reward)):
    ts_hddpg_reward[i] = ts_hddpg_reward[i]/((i+1)*50)
print(ts_hddpg_reward[-1])
plt.plot(np.arange(0, len(ts_hddpg_reward)), ts_hddpg_reward, label='ts_hddpg')

plt.legend()
plt.show()

from matplotlib.backends.backend_pdf import PdfPages
# plt.savefig('./simulator_reward_criteo.pdf')