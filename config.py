import tensorflow as tf
import numpy as np

np.random.seed(1)
tf.set_random_seed(1)

#####################  Data hyper parameters  ####################
# dataset size
data_size =  10763
canditate_size = 16

MAX_EPISODES = data_size
MAX_EP_STEPS = 200

file_path = ''

###############################  Meta controller hyper parameters(HLA)  ##############################
LR_A = 0.0005    # learning rate for actor
LR_C = 0.0005    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement

MEMORY_CAPACITY = 500 #1000 #pvpre 400
BATCH_SIZE = 64 #128 #pvpre 48

# RENDER = False
OUTPUT_GRAPH = True

EPSILON = 1  # control exploration

state_dim = 28*16
item_dim = 28
action_dim = 28 #28(item预测) 28*6(pv预测)
action_bound = 1 # discard

rnn_hidden_dim = 32
###############################  Controller hyper parameters(LLA)  ##############################

c_LR_A = 0.0001    # learning rate for actor
c_LR_C = 0.0001    # learning rate for critic
c_GAMMA = 0.9     # reward discount
c_TAU = 0.01      # soft replacement

c_MEMORY_CAPACITY = 1000 # 500
c_BATCH_SIZE = 64 # 64

c_EPSILON = 1  # control exploration

c_state_dim = 28*12 #28*6+28*6 #pvpre 28*16+28*6+28*6
c_action_dim = 28
c_action_bound = 1 # discard

c_reward_alpha = 1 # reward = å ndcg + (1-å) gmv

dynamic_reward_flag = True # dynamic ndcg or not

c_rnn_hidden_dim = 64
#####################  Cluster hyper parameters  ####################
INCRE_update = True
INCRE_steps = [] # dynamic incremental update

#####################  train|test hyper parameters  ####################
split_rate = 0.8 # training size rate