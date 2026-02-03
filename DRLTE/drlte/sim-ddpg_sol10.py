#-*-coding:UTF-8 -*-
from socket import *
import datetime
import os
from os import path as osp
import sys
import math
import tensorflow as tf
import numpy as np
import utilize
from utilize import clock
from Network.actor import ActorNetwork
from Network.critic import CriticNetwork
from ReplayBuffer.replaybuffer import PrioritizedReplayBuffer, ReplayBuffer
from Summary.summary import Summary
from Explorer.explorer import Explorer
from SimEnv.Env1110 import Env
from flag import FLAGS
import json
import timeit
import time

from tensorflow.compat.v1.logging import log, log_if, log_first_n, set_verbosity
from tensorflow.compat.v1.logging import INFO, ERROR, DEBUG, WARN

set_verbosity(INFO)
"""读入参数 及 重新设定参数 """
if not hasattr(sys, 'argv'):
    sys.argv = ['']

######hyper
TIME_STAMP = str(datetime.datetime.now())
SERVER_PORT = getattr(FLAGS, 'server_port')
SERVER_IP = getattr(FLAGS, 'server_ip')

OFFLINE_FLAG = getattr(FLAGS, 'offline_flag')
ACT_FLAG = getattr(FLAGS, 'act_flag')
SEED = getattr(FLAGS, 'random_seed')

ACTOR_LEARNING_RATE = getattr(FLAGS, 'learning_rate_actor')
CRITIC_LEARNING_RATE = getattr(FLAGS, 'learning_rate_critic')

GAMMA = getattr(FLAGS, 'gamma')
TAU = getattr(FLAGS, 'tau')
ALPHA = getattr(FLAGS, 'alpha')
BETA = getattr(FLAGS, 'beta')
MU = getattr(FLAGS, 'mu')
DELTA = getattr(FLAGS, 'delta')

EP_BEGIN = getattr(FLAGS, 'epsilon_begin')
EP_END = getattr(FLAGS, 'epsilon_end')
EP_ST = getattr(FLAGS, 'epsilon_steps')

ACTION_BOUND = getattr(FLAGS, 'action_bound')

BUFFER_SIZE = getattr(FLAGS, 'size_buffer')
MINI_BATCH = getattr(FLAGS, 'mini_batch')

MAX_EPISODES = getattr(FLAGS, 'episodes')
MAX_EP_STEPS = getattr(FLAGS, 'epochs')

if getattr(FLAGS, 'stamp_type') == '__TIME_STAMP':
    REAL_STAMP = TIME_STAMP
else:
    REAL_STAMP = getattr(FLAGS, 'stamp_type')
DIR_SUM = getattr(FLAGS, 'dir_sum').format(REAL_STAMP)
DIR_RAW = getattr(FLAGS, 'dir_raw').format(REAL_STAMP)
DIR_MOD = getattr(FLAGS, 'dir_mod').format(REAL_STAMP)
DIR_LOG = getattr(FLAGS, 'dir_log').format(REAL_STAMP)
DIR_CKPOINT = getattr(FLAGS, 'dir_ckpoint').format(REAL_STAMP)

AGENT_TYPE = getattr(FLAGS, "agent_type")

DETA_W = getattr(FLAGS, "deta_w")
DETA_L = getattr(FLAGS, "deta_l")  # for multiagent deta_w < deta_l

EXP_EPOCH = getattr(FLAGS, "explore_epochs")
EXP_DEC = getattr(FLAGS, "explore_decay")

CKPT_PATH = getattr(FLAGS, "ckpt_path")
start_step = 0
INFILE_NAME = getattr(FLAGS, "file_name")
INFILE_PATHPRE = "../inputs/"
assert INFILE_NAME != "", "INFILE NAME error"
INFILE_PATH = osp.join(INFILE_PATHPRE, INFILE_NAME + ".txt")
# hesy add for metric of performance ratio
path_lp_perform = getattr(FLAGS, 'lpPerformFile')

IS_TRAIN = getattr(FLAGS, "is_train")
START_INDEX = getattr(FLAGS, "train_start_index")
STATE_NORMAL = getattr(FLAGS, "state_normal")

# add for new training method
tm_circle = getattr(FLAGS, "tm_circle")
len_circle = getattr(FLAGS, "len_circle")

######hyper


#(_g represent global state or joint action)
class DrlAgent:
    def __init__(self, state_init, action_init, state_init_g, action_init_g, dim_state, dim_action, dim_state_g, dim_action_g, act_ind, num_paths, exp_action, sess):  #=
        self.__dim_state = dim_state  #=
        self.__dim_action = dim_action  #=
        #added by fei
        self.__dim_state_g = dim_state_g
        self.__dim_action_g = dim_action_g
        ###
        self.__actor = ActorNetwork(sess, dim_state, dim_action, ACTION_BOUND, ACTOR_LEARNING_RATE, TAU, num_paths)
        self.__critic = CriticNetwork(sess, dim_state, dim_action, dim_state_g, dim_action_g, act_ind, CRITIC_LEARNING_RATE, TAU, self.__actor.num_trainable_vars)

        self.__prioritized_replay = PrioritizedReplayBuffer(BUFFER_SIZE, MINI_BATCH, ALPHA, MU, SEED)
        self.__replay = ReplayBuffer(BUFFER_SIZE, SEED)  # being deprecated ?

        self.__explorer = Explorer(EP_BEGIN, EP_END, EP_ST, dim_action, num_paths, SEED, exp_action, EXP_EPOCH, EXP_DEC)

        self.__state_curt = state_init
        self.__action_curt = action_init
        #added by fei
        self.__state_curt_g = state_init_g
        self.__action_curt_g = action_init_g
        ###
        self.__base_sol = utilize.get_base_solution(dim_action)  # depretated

        self.__episode = 0
        self.__step = 0
        self.__ep_reward = 0.
        self.__ep_avg_max_q = 0.

        self.__beta = BETA

        self.__detaw = DETA_W
        self.__detal = DETA_L

        self.__maxutil = 100.

    def target_paras_init(self):
        # can be modified
        self.__actor.update_target_paras()
        self.__critic.update_target_paras()

    @property
    def timer(self):
        return '| %s '%ACT_FLAG \
               + '| tm: %s '%datetime.datetime.now() \
               + '| ep: %.4d '%self.__episode \
               + '| st: %.4d '%self.__step

    def pre_predict(self, state_new, reward, maxutil):  #=== state_local + state_global
        """ 两个actor网络根据转移到的新状态做出action
        Args:
            state_new : 当前智能体起始的session的备选路径上所有经过的link的链路利用率
            reward : 当前智能体获得的reward
            maxutil : 全局拓扑最大链路利用率大小
        Returns:
            action_target : target actor network针对state_new做出的action选择
            action : actor network针对state_new做出的action选择
        """
        if self.__maxutil > maxutil:
            self.__maxutil = maxutil
            self.__explorer.setExpaction(self.__action_curt)

        self.__step += 1
        self.__ep_reward += reward

        if self.__step >= MAX_EP_STEPS:
            self.__step = 0
            self.__episode += 1
            self.__ep_reward = 0.
            self.__ep_avg_max_q = 0.
            if OFFLINE_FLAG:
                self.__explorer.setEp(EP_BEGIN * 0.625)
                self.__explorer.setExpaction(self.__action_curt)
                self.__maxutil = 100.

        #return a array
        action_target = self.__actor.predict_target([state_new])[0]
        #return a array
        action_original = self.__actor.predict([state_new])[0]
        #return a list
        action = self.__explorer.get_act(action_original, self.__episode, flag=ACT_FLAG)

        return action_target, action

    @clock
    def predict(self, state_new, state_new_g, action_g_t, action_g, action, reward, act_ind, dim_a):
        """计算下一步要执行的action以及target_q用于计算loss更新网络参数
        _g represents global action, _t is target
        Args:
            state_new : 当前智能体起始的session的备选路径上所有经过的link的链路利用率 (list，元素个数是sessNum*pathNum*hopNum )
            state_new_g : 所有智能体的state_new
            action_g_t : 所有智能体的target actor network的action
            action_g : 所有智能体的actor network的action
            action : 该智能体的actor network的action
            reward : 该智能体的reward
            act_ind : 该智能体动作在全局动作的起始index
            dim_a : 该智能体动作的维度
        Returns:
           action : 下一步实际要执行的action
           target_q : reward + Q(s',a')
        """
        #=== state_local + state_global
        ###################################################################
        # _g represents global action, _t is traget
        # action_g   represents the
        # action_g_t represents
        ###################################################################

        target_q = 0.0  #for return value in reference
        # Priority
        if IS_TRAIN:
            target_q = self.__critic.predict_target(  #===
                [state_new_g], [action_g_t])[0]
            value_q = self.__critic.predict([self.__state_curt_g], [self.__action_curt_g])[0]  #==
            #grads = self.__critic.calculate_gradients([self.__state_curt_g], [self.__action_curt[act_ind : act_ind+dim_a]]) #==

            _grads = self.__critic.calculate_gradients([self.__state_curt_g], [self.__action_curt_g])[0]
            grads = _grads[:, act_ind:act_ind + dim_a]

            td_error = abs(reward + GAMMA * target_q - value_q)

            transition = (self.__state_curt, self.__action_curt, reward, state_new, self.__state_curt_g, self.__action_curt_g, state_new_g)
            self.__prioritized_replay.add(transition, td_error, [np.mean(np.abs(grads))])
            #self.__replay.add(transition[0], transition[1], transition[2], transition[3])

        self.__state_curt = state_new  #===
        self.__action_curt = action  #===
        ### added by fei
        self.__state_curt_g = state_new_g
        self.__action_curt_g = action_g
        ###

        return action, target_q  # target_q fot post()

    def post(self, state_new_g, action_g, batch_act_t_train_g, batch_act_train_g, act_ind, dim_act, batch, weights, indices, target_q):
        """根据pre_train获取的五元组，调用train函数计算用于更新的梯度，各智能体独立更新,更新集中critic

        Args:
            state_new_g : 所有智能体的state_new
            action_g : 所有智能体的actor network的action
            batch_act_t_train_g : 在批数据中，所有智能体针对各自的state_new，各自的target actor网络做出的对应的决策的集合
            batch_act_train_g : 在批数据中，所有智能体针对各自的state_new，各自的actor网络做出的对应的决策的集合
            act_ind : 该智能体动作在全局动作的起始index
            dim_act : 该智能体动作的维度
            batch : 从buffer中取出的批数据
            weights : 批数据中各样本的权重
            indices : 批数据中各样本在buffer中的序号
            target_q : reward + Q(s',a')
        """

        if len(self.__prioritized_replay) > MINI_BATCH and IS_TRAIN:
            curr_q = self.__critic.predict_target([state_new_g], [action_g])[0]
            if curr_q[0] > target_q[0]:  #? ? ? ?
                self.train(True, batch_act_t_train_g, batch_act_train_g, act_ind, dim_act, batch, weights, indices)  #=
            else:
                self.train(False, batch_act_t_train_g, batch_act_train_g, act_ind, dim_act, batch, weights, indices)  #=

    def pre_train(self):
        """如果buffer中数据量够且处于训练模式下，采样batch的同时获取他们的辅助信息（样本权重，样本序号），同时针对取出数据中的state_new计算出此刻actor网络和target actor网络对应的action

        Returns:
            batch_action_t_train: 根据buffer中取出来的新的state,target actor network做出的action
            batch_action_train : 根据buffer中取出来的新的state,actor network做出的action
            batch : 取出的五元组数据
            weights : 样本的权重
            indices : 样本在buffer中对应的序号 
            Is_enter : 是否要继续下一步的神经网络参数更新
        """
        if len(self.__prioritized_replay) > MINI_BATCH and IS_TRAIN:
            self.__beta += (1 - self.__beta) / EP_ST

            batch, weights, indices = self.__prioritized_replay.select(self.__beta)
            weights = np.expand_dims(weights, axis=1)

            batch_state = []
            batch_state_next = []
            for val in batch:
                try:
                    batch_state.append(val[0])
                    batch_state_next.append(val[3])
                except TypeError:
                    print('*' * 20)
                    print('--val--', val)
                    print('*' * 20)
                    continue

            batch_action_t_train = self.__actor.predict_target(batch_state_next)
            batch_action_train = self.__actor.predict(batch_state)

            Is_enter = True
            return batch_action_t_train, batch_action_train, batch, weights, indices, Is_enter

        Is_enter = False
        return None, None, None, None, None, Is_enter

    def train(self, curr_stat, batch_act_t_train_g, batch_act_train_g, act_ind, dim_act, batch, weights, indices):
        """根据样本数据，计算出target Q值，更新actor网络，再进一步计算出TD error，更新actor网络

        Args:
            curr_stat : true意味着智能体之前的更新方向比较正确，所以学习率小一点。false意味着智能体之前的更新方向偏差较大，所以学习率大一点，努力追赶。
            batch_act_t_train_g : 在批数据中，所有智能体针对各自的state_new，各自的target actor网络做出的对应的决策的集合
            batch_act_train_g : 在批数据中，所有智能体针对各自的state_new，各自的actor网络做出的对应的决策的集合
            act_ind : 该智能体动作在全局动作的起始index
            dim_act : 该智能体动作的维度
            batch : 从buffer中取出的批数据
            weights : 批数据中各样本的权重
            indices : 批数据中各样本在buffer中的序号
        """
        batch_state = []
        batch_action = []
        batch_reward = []
        batch_state_next = []
        batch_state_g = []
        batch_action_g = []
        batch_state_next_g = []
        for val in batch:
            try:  #===
                batch_state.append(val[0])
                batch_action.append(val[1])
                batch_reward.append(val[2])
                batch_state_next.append(val[3])

                batch_state_g.append(val[4])
                batch_action_g.append(val[5])
                batch_state_next_g.append(val[6])
            except TypeError:
                print('*' * 20)
                print('--val--', val)
                print('*' * 20)
                continue

        target_q = self.__critic.predict_target(batch_state_next_g, batch_act_t_train_g)  #=
        value_q = self.__critic.predict(batch_state_g, batch_action_g)

        batch_y = []
        batch_error = []
        for k in range(len(batch_reward)):
            target_y = batch_reward[k] + GAMMA * target_q[k]
            batch_error.append(abs(target_y - value_q[k]))
            batch_y.append(target_y)

        predicted_q, _ = self.__critic.train(batch_state_g, batch_action_g, batch_y, weights)  #===

        self.__ep_avg_max_q += np.amax(predicted_q)

        #a_outs = self.__actor.predict(batch_state)
        _grads = self.__critic.calculate_gradients(batch_state_g, batch_act_train_g)[0]  #=
        grads = _grads[:, act_ind:act_ind + dim_act]

        # Prioritized
        self.__prioritized_replay.priority_update(indices, np.array(batch_error).flatten(), np.mean(np.abs(grads), axis=1))
        weighted_grads = weights * grads

        if curr_stat:
            weighted_grads *= self.__detaw
        else:
            weighted_grads *= self.__detal
        self.__actor.train(batch_state, weighted_grads)

        self.__actor.update_target_paras()
        self.__critic.update_target_paras()


'''initial part'''
log(DEBUG, "\n----Information list----")
log(DEBUG, "agent_type: %s" % (AGENT_TYPE))
log(DEBUG, "stamp_type: %s" % (REAL_STAMP))
UPDATE_TIMES = 0

env = Env(INFILE_PATHPRE, INFILE_NAME, INFILE_NAME.split("_")[0], MAX_EP_STEPS, SEED, START_INDEX, tm_circle, len_circle, start_step)
# alternative_path , topofile , topo , epochs ,

# env.showInfo()    # get initial info
NODE_NUM, SESS_NUM_ORG, EDGE_NUM, NUM_PATHS_ORG, SESS_PATHS_ORG, EDGE_MAP = env.getInfo()  # SESS_PATHS shows the nodes in each path of each session

# Remove single-path session
SESS_NUM = 0  # 备选路径的总数量; 备选路径大于1(需要进行流量分割)的session对的数量
NUM_PATHS = []  # ( ODPairNum, ) 备选路径数量大于1的session对的备选路径数量
SESS_PATHS = []  # ( ODPairNum, SessionPathNum, EachHopOnPath ) 备选路径数量大于1的session对的备选路径
for i in range(SESS_NUM_ORG):
    if NUM_PATHS_ORG[i] == 1:
        continue
    SESS_NUM += 1
    NUM_PATHS.append(NUM_PATHS_ORG[i])
    SESS_PATHS.append(SESS_PATHS_ORG[i])
sess_src = [SESS_PATHS[sessIdx][0][0] for sessIdx in range(SESS_NUM)]  # 数量级：ODPair*ODPair
DIM_ACTION = sum(NUM_PATHS)  # 单条备选路径不需要进行流量分割

agents = []
# init routing/scheduling policy: multi_agent, drl_te, mcf, ospf, .etc

if AGENT_TYPE == "multi_agent":
    # get the action path initially set
    action_path = getattr(FLAGS, "action_path")
    if action_path != None:
        action_ori = []
        action_file = open(action_path, 'r')
        for i in action_file.readlines():
            action_ori.append(float(i.strip()))
        ind = 0
        action = []
        for i in range(SESS_NUM_ORG):
            if NUM_PATHS_ORG[i] > 1:
                action += action_ori[ind:ind + NUM_PATHS_ORG[i]]
            ind += NUM_PATHS_ORG[i]
    else:
        action = utilize.convert_action(np.ones(DIM_ACTION), NUM_PATHS)  # !

    AGENT_NUM = max(sess_src) + 1  # here AGENT_NUM is not equal to the real valid "agent number"
    srcSessNum = [0] * AGENT_NUM  # nodeIdx为源的sess的数量有是如此srcSessNum[nodeIdx]个
    srcPathNum = [0] * AGENT_NUM  # nodeIdx为源的sess的所有备选路径数量总和是srcPathNum[nodeIdx]个 (无论sess的目的节点是哪个)
    srcUtilNum = [0] * AGENT_NUM  # sum util num for each src (sum util for each path)
    srcPaths = [[] for i in range(AGENT_NUM)]
    srcActs = [[] for i in range(AGENT_NUM)]  # the initial explore actions for the agents

    actp = 0
    for i in range(len(sess_src)):
        srcSessNum[sess_src[i]] += 1
        srcPathNum[sess_src[i]] += NUM_PATHS[i]
        srcPaths[sess_src[i]].append(NUM_PATHS[i])  # （ODPair,[PathsSet1,PathsSet2,...]）
        srcActs[sess_src[i]] += action[actp:actp + NUM_PATHS[i]]
        actp += NUM_PATHS[i]
    #calculate srcUtilNum(unique)
    sess_util_tmp = [{} for i in range(AGENT_NUM)]  # 记录了每个agent的所有备选路径的hop组合在一起的set
    for sessIdx in range(SESS_NUM):
        for pathIdx in range(NUM_PATHS[sessIdx]):
            for hopIdx in range(len(SESS_PATHS[sessIdx][pathIdx]) - 1):
                enode1 = SESS_PATHS[sessIdx][pathIdx][hopIdx]
                enode2 = SESS_PATHS[sessIdx][pathIdx][hopIdx + 1]
                id_tmp = str(enode1) + "," + str(enode2)
                if id_tmp not in sess_util_tmp[sess_src[sessIdx]]:
                    sess_util_tmp[sess_src[sessIdx]][id_tmp] = 0

    # srcUtilNum[agentIdx] 第agentIdx个src的所有备选路径的hop的数量
    for i in range(AGENT_NUM):
        srcUtilNum[i] = len(sess_util_tmp[i].values())

    ### added by fei
    state_global = []
    action_global = []
    dim_s_global = 0
    dim_a_global = 0
    act_ind = []
    p = 0

    dirLinkNum = [0 for _ in range(NODE_NUM)]
    for fromNode in range(NODE_NUM):
        for toNode in range(NODE_NUM):
            if fromNode == toNode or not EDGE_MAP[fromNode][toNode]: continue
            dirLinkNum[fromNode] += 1
    log(INFO, f"dirLinkNum is {dirLinkNum}")

    for i in range(AGENT_NUM):
        state = np.zeros(NODE_NUM - 1 + dirLinkNum[i])  # cernet is (n-1)*n
        # state = np.zeros(NODE_NUM-1)    # abi is n*n
        state_global += list(state)
        action = utilize.convert_action(np.ones(srcPathNum[i]), srcPaths[i])
        action_global += action
        dim_s_global += NODE_NUM - 1 + dirLinkNum[i]
        act_ind.append(dim_a_global)  #store the starting index in joint action list for agent i
        dim_a_global += srcPathNum[i]

    log(DEBUG, "***********Global state and action information************")
    log(DEBUG, "global state dimension is(fei): ", dim_s_global)
    log(DEBUG, "global action dimension is(fei): ", dim_a_global)

    ###
    # construct the distributed agents
    log(DEBUG, "\nConstructing distributed agents ... \n")
    #* hesy add
    gpu_options = tf.GPUOptions(allow_growth=True)
    globalSess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    for i in range(AGENT_NUM):
        log(DEBUG, "agent %d" % i)
        if (srcSessNum[i] > 0):  #valid agent
            temp_dim_s = NODE_NUM - 1 + dirLinkNum[i]  # one tm is n*(n-1)
            if not STATE_NORMAL:
                temp_dim_s += 2
            state = np.zeros(temp_dim_s)
            action = utilize.convert_action(np.ones(srcPathNum[i]), srcPaths[i])
            agent = DrlAgent(
                list(state),
                action,
                state_global,
                action_global,  # state and action
                temp_dim_s,
                srcPathNum[i],
                dim_s_global,
                dim_a_global,  #s_dim and a_dim
                act_ind[i],
                srcPaths[i],
                srcActs[i],
                globalSess)
        else:
            agent = None
        agents.append(agent)

    # parameters init
    log(DEBUG, "Running global_variables initializer ...")
    globalSess.run(tf.global_variables_initializer())

    # init target actor and critic para
    log(DEBUG, "Building target network ...")
    for i in range(AGENT_NUM):
        if agents[i] != None:  #valid agent
            agents[i].target_paras_init()

    # parameters restore
    mSaver = tf.train.Saver(tf.trainable_variables())
    if CKPT_PATH != None:
        log(INFO, f"restoring paramaters from {CKPT_PATH}")
        mSaver.restore(globalSess, CKPT_PATH)
        start_step = int(CKPT_PATH.split('-')[-1]) if IS_TRAIN else 0
        log(INFO, f"start_step is {start_step}")

    ret_c = tuple(action)  # deprecated now

if not os.path.exists(DIR_LOG):
    os.makedirs(DIR_LOG)
if not os.path.exists(DIR_CKPOINT):
    os.makedirs(DIR_CKPOINT)

# file_sta_out = open(DIR_LOG + '/sta.log', 'w', 1)
file_rwd_out = open(DIR_LOG + '/rwd.log', 'w', 1)
# file_act_out = open(DIR_LOG + '/act.log', 'w', 1)
file_perfm = open(DIR_LOG + '/perfm.log', 'w', 1)
file_util_out = open(DIR_LOG + '/util.log', 'w', 1)  # file record the max util
# file_multirwd_out = open(DIR_LOG + '/multirwd.log', 'w', 1)   # record the rwd for each agent

#Fei Gui
_reward = []
_max_util = []


########
def split_arg(max_util, sess_path_util, net_util, lastTM):  # start_step is global
    # def split_arg(max_util, sess_path_util, net_util):  # start_step is global
    """处理environment给出的信息，解析出state

    Args:
        max_util (float): 全网最大链路利用率
        sess_path_util : 备选路径上每一个链路的链路利用率大小 (ODPairNum,SessionPathNum,Link Util aLong Each Session Path)
        net_util : 拓扑上各链路的链路利用率
        lastTM : 上一时刻的TM数据

    Returns:
        multi_state_new : 各智能体观察到的新的状态 ( nodeNum , sessNum*pathNum*hopNum )
        state_global : 相当于ravel了multi_state_new ( nodeNum* sessNum*pathNum*hopNum, )
        multi_reward : 各智能体获得的奖励值 (nodeNum ,)
    """
    global _reward
    #************* hesy add
    if not hasattr(split_arg, "lp_perf"):  # lp_perf: MLU under optimal solution (calculated by linear programming)
        split_arg.lp_perf = []
        try:
            with open(path_lp_perform, "r") as f:
                split_arg.lp_perf = np.loadtxt(f)  # 格式

        except OSError as reason:
            log(ERROR, f"{path_lp_perform} not exists")
            exit(-1)

        log(INFO, f"read in split_arg.lp_perf from file:{path_lp_perform}")

    if not hasattr(split_arg, "episode"):
        split_arg.episode = 0
        log(INFO, f"initialize split_arg.episode to be {split_arg.episode}")

    if not hasattr(split_arg, "updateTime"):
        split_arg.updateTime = start_step + 1 if start_step else 0
        log(INFO, f"initialize split_arg.updateTime to be {split_arg.updateTime}")
    else:
        split_arg.updateTime += 1

    if split_arg.updateTime % MAX_EP_STEPS == 0 and split_arg.updateTime != 0:
        split_arg.episode += 1
        # logger.info(f"update split_arg.episode to be {split_arg.episode}")
        log(INFO, f"update split_arg.episode to be {split_arg.episode}")
    #*************

    print(max_util, file=file_util_out)

    if AGENT_TYPE == "multi_agent":
        maxsess_util = [[] for i in range(AGENT_NUM)]  # (nodeNum, sessNum*PathNum ) max link utilization for each candidate path for sessions origining from each agent

        # get maxsess_util
        for sessIdx in range(SESS_NUM):
            temp_sessmax = 0.
            for pathIdx in sess_path_util[sessIdx]:
                temp_sessmax = max(temp_sessmax, max(pathIdx))
            maxsess_util[sess_src[sessIdx]].append(temp_sessmax)

        # calculate state_new, and reward for each agent
        multi_state_new = []  #? check new # 各智能体观察到的新的状态
        state_global = []  # ravel multi_state_new
        multi_reward = []  # 各智能获得的reward
        tm_index = split_arg.updateTime // (tm_circle * len_circle) * tm_circle + split_arg.updateTime % tm_circle
        tm_index = tm_index % len(split_arg.lp_perf)
        log_first_n(INFO, f"tm_index,len(split_arg.lp_perf) is {tm_index} and {len(split_arg.lp_perf)} ,while its relative lp is {split_arg.lp_perf[tm_index]}", 3)

        maxTM = max(lastTM)
        for i in range(AGENT_NUM):
            if (agents[i] == None):
                state_new = None
                reward = None
            else:
                if not STATE_NORMAL:
                    state_new = net_util[i] + [max_util]
                    state_new.extend(lastTM[i * (NODE_NUM - 1):(i + 1) * (NODE_NUM - 1)] + [maxTM])  # one tm is (n-1)*n
                else:
                    state_new = list(np.array(net_util[i]) / max_util)
                    state_new.extend(list(np.array(lastTM[i * (NODE_NUM - 1):(i + 1) * (NODE_NUM - 1)]) / maxTM))
                reward = (-0.05 * (np.mean(maxsess_util[i]) + DELTA / len(maxsess_util[i]) * max_util)) / split_arg.lp_perf[tm_index]

            multi_state_new.append(state_new)
            state_global += state_new  ### added by fei
            multi_reward.append(reward)

        Pf_Ratio = (1.0 * max_util) / split_arg.lp_perf[tm_index]
        print(Pf_Ratio, file=file_perfm)
        _reward.append(sum(multi_reward))

        return multi_state_new, state_global, multi_reward


# def step(max_util, sess_path_util, net_util):
def step(max_util, sess_path_util, net_util, lastTM):
    """根据流模拟器给的网络状态信息，解析出state，训练各agent并获取下一步的action

    Args:
        max_util (float): 全网最大链路利用率
        sess_path_util : 备选路径上每一个链路的链路利用率大小 (ODPairNum,SessionPathNum,Link Util aLong Each Session Path)
        net_util :  拓扑上各链路的链路利用率
        lastTM : 上一时刻的TM数据

    Returns:
        ret_c: 用于下一个TM使用的action
    """
    state, state_global, reward = split_arg(max_util, sess_path_util, net_util, lastTM)
    ### added by fei
    ### obtain joint action by predicting for each agent
    action_target_global = np.array([], dtype="float32")  # 聚合所有agent的target network的action
    act_local = []  # 聚合所有agent的online network的action
    action_global = []  # 相当于ravel了act_local
    for agentIdx in range(AGENT_NUM):
        ##################################################################################
        start_time = time.time()
        _act_target, _act = agents[agentIdx].pre_predict(state[agentIdx], reward[agentIdx], max_util)
        # print("pre_predict inference time is: ", 1000 * (time.time() - start_time))

        action_target_global = np.append(action_target_global, _act_target)
        action_global += _act

        act_local.append(_act)
        # return value:
        # action_target_global is a array shape: [370, ]
        # action_global is a list, shape: [370, ]
        ###

    dim_act = []
    target_q = []
    if AGENT_TYPE == "multi_agent":
        ret_c_t = []
        ret_c = []
        for i in range(AGENT_NUM):
            ###################################################################################
            if agents[i] != None:
                dim_act.append(srcPathNum[i])  # the action dimensions of agent i
                # rsult is a list
                t0 = time.time()
                result, _target_q, elapsed = agents[i].predict(state[i], state_global, action_target_global, action_global, act_local[i], reward[i], act_ind[i], dim_act[i])
                # print("verified time is : ", 1000 * (time.time() - t0))

                ret_c_t.append(result)
                target_q.append(_target_q)
                log_first_n(INFO, f"the inference time is: {elapsed}", 5)
            else:
                ret_c_t.append([])
        for i in range(len(sess_src)):
            ret_c += ret_c_t[sess_src[i]][0:NUM_PATHS[i]]
            ret_c_t[sess_src[i]] = ret_c_t[sess_src[i]][NUM_PATHS[i]:]

        batch_action_t_train_g = np.array([[] for i in range(MINI_BATCH)], dtype="float32")  # global batch action from actor.predict_target()
        batch_action_train_g = np.array([[] for i in range(MINI_BATCH)], dtype="float32")  # global batch action from actor.predict()
        batch_n = []
        weights_n = []
        indices_n = []
        Is_To_train = False
        for i in range(AGENT_NUM):
            ###################################################################################
            _act_batch_t, _act_batch, batch, weights, indices, Is_enter = agents[i].pre_train()
            if Is_enter == True:
                Is_To_train = True
                batch_action_t_train_g = np.append(batch_action_t_train_g, _act_batch_t, axis=1)  #!!!!!!!!!!!!!
                batch_action_train_g = np.append(batch_action_train_g, _act_batch, axis=1)  #!!!!!!!!!!!!!
                batch_n.append(batch)
                weights_n.append(weights)
                indices_n.append(indices)
        ### Return value:
        # batch_action_t_train_g is a array, shape: [32, 370]
        # batch_action_train_g is a array,shape: [32, 370]
        ###################################################################################
        if Is_To_train == True:
            for i in range(AGENT_NUM):
                agents[i].post(state_global, action_global, batch_action_t_train_g, batch_action_train_g, act_ind[i], dim_act[i], batch_n[i], weights_n[i], indices_n[i], target_q[i])
    reward_t = 0.
    for i in reward:
        if i != None:
            # for multiagent each agent has a reward
            reward_t += i

    print(reward_t, file=file_rwd_out)
    # print(reward, file=file_multirwd_out)
    return ret_c


if __name__ == "__main__":
    #fei
    time_start = time.clock()
    last_time = time_start
    UPDATE_TIMES = start_step
    route = []
    if OFFLINE_FLAG:
        log(INFO, f"begin training from step {start_step}")
        for cur_step in range(start_step, MAX_EPISODES * MAX_EP_STEPS):
            log_first_n(INFO, f"elapsed time for epoch {cur_step} is {time.clock()-last_time}", 100)
            last_time = time.clock()
            log_if(INFO, f"************* episode {cur_step//MAX_EP_STEPS}  *************", cur_step % MAX_EP_STEPS == 0)
            UPDATE_TIMES += 1

            # max_util, sess_path_util_org, net_util = env.update(route)
            max_util, sess_path_util_org, net_util, lastTM = env.update_sol10(route)
            #plot reward , added by fei
            _max_util.append(max_util)
            ############
            sess_path_util = []
            for j in range(SESS_NUM_ORG):
                if NUM_PATHS_ORG[j] > 1:
                    sess_path_util.append(sess_path_util_org[j])

            ret_c = step(max_util, sess_path_util, net_util, lastTM)

            ret_c_count = 0
            route = []
            for j in range(SESS_NUM_ORG):
                if NUM_PATHS_ORG[j] > 1:
                    for k in range(NUM_PATHS_ORG[j]):
                        route.append(round(ret_c[ret_c_count], 3))
                        ret_c_count += 1
                else:
                    route.append(1.0)
            # print(route, file=file_act_out)

            # store global variables
            save_step = 10000
            if (AGENT_TYPE == "multi_agent" or AGENT_TYPE == "drl_te") and IS_TRAIN and cur_step % save_step == 0:
                log(INFO, f"At step {cur_step} saves checkpoint at {DIR_CKPOINT + '/ckpt'}")
                mSaver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
                mSaver.save(globalSess, DIR_CKPOINT + "/ckpt", global_step=cur_step, write_meta_graph=cur_step == 0)  # save graph only at the first time

    time_elapse = (time.clock() - time_start)
    log_first_n(INFO, "elasped {time_elapse} s", 100)