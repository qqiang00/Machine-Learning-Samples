import random

import numpy as np
import math


def sigmoid(x):
  return 1. / (1 + np.exp(-x))


# createst uniform random array w/ values in [a,b) and shape args
# 返回一个随机数矩阵，矩阵内元素的值范围在(a,b)之间，矩阵的shape由*args确定
def rand_arr(a, b, *args):
  np.random.seed(0)
  return np.random.rand(*args) * (b - a) + a

# 用来管理LSTM参数的类
# mem_cell_ct指LSTM内状态数量
# x_dim接受输入数据的维度
class LstmParam:
  def __init__(self, mem_cell_ct, x_dim):
    # 接受参数给自身的变量
    self.mem_cell_ct = mem_cell_ct
    self.x_dim = x_dim
    # 合并细胞状态数和输入数
    # 因为在LSTM内部的计算过程中，细胞前t-1的状态和输入序列都参与确定LSTM的输出和t时刻细胞状态
    concat_len = x_dim + mem_cell_ct
    # weight matrices
    # wg:输入层 -> LSTM的连接权重，shape=[concat_len,mem_cell_ct]
    self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
    # wi:LSTM内输入门权重矩阵
    self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
    # wf:LSTM内忘记门权重矩阵
    self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
    # wo:LSTM内输出门权重矩阵
    self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
    # bias terms
    # 相关的偏置项bias
    self.bg = rand_arr(-0.1, 0.1, mem_cell_ct)
    self.bi = rand_arr(-0.1, 0.1, mem_cell_ct)
    self.bf = rand_arr(-0.1, 0.1, mem_cell_ct)
    self.bo = rand_arr(-0.1, 0.1, mem_cell_ct)

    # diffs (derivative of loss function w.r.t. all parameters)
    # 各矩阵的偏微分矩阵(权重和偏置项更新矩阵)
    self.wg_diff = np.zeros((mem_cell_ct, concat_len))
    self.wi_diff = np.zeros((mem_cell_ct, concat_len))
    self.wf_diff = np.zeros((mem_cell_ct, concat_len))
    self.wo_diff = np.zeros((mem_cell_ct, concat_len))
    self.bg_diff = np.zeros(mem_cell_ct)
    self.bi_diff = np.zeros(mem_cell_ct)
    self.bf_diff = np.zeros(mem_cell_ct)
    self.bo_diff = np.zeros(mem_cell_ct)

  # 更新权重，重置更新矩阵
  def apply_diff(self, lr=1):
    self.wg -= lr * self.wg_diff
    self.wi -= lr * self.wi_diff
    self.wf -= lr * self.wf_diff
    self.wo -= lr * self.wo_diff
    self.bg -= lr * self.bg_diff
    self.bi -= lr * self.bi_diff
    self.bf -= lr * self.bf_diff
    self.bo -= lr * self.bo_diff
    # reset diffs to zero
    self.wg_diff = np.zeros_like(self.wg)
    self.wi_diff = np.zeros_like(self.wi)
    self.wf_diff = np.zeros_like(self.wf)
    self.wo_diff = np.zeros_like(self.wo)
    self.bg_diff = np.zeros_like(self.bg)
    self.bi_diff = np.zeros_like(self.bi)
    self.bf_diff = np.zeros_like(self.bf)
    self.bo_diff = np.zeros_like(self.bo)

# LSTM状态类
# 记录LSTM内自身以及各门的状态
class LstmState:
  def __init__(self, mem_cell_ct, x_dim):
    # 矩阵g存放的是经过tanh门后、未与输入门状态进行乘积前的状态
    self.g = np.zeros(mem_cell_ct)
    # 矩阵i存放经过输入门后的状态
    self.i = np.zeros(mem_cell_ct)
    # 矩阵f存放经过忘记门后的状态
    self.f = np.zeros(mem_cell_ct)
    # 矩阵o存放经过书出门后的状态
    self.o = np.zeros(mem_cell_ct)
    # 矩阵s存放LSTM内自身记忆细胞的状态
    self.s = np.zeros(mem_cell_ct)
    # 矩阵h存放的是LSTM输出，相当于隐藏层的输出
    self.h = np.zeros(mem_cell_ct)

    self.bottom_diff_h = np.zeros_like(self.h)
    self.bottom_diff_s = np.zeros_like(self.s)
    self.bottom_diff_x = np.zeros(x_dim)

# LSTM节点类
class LstmNode:
  # 用LSTM参数和LSTM状态类记录描述LSTM节点
  def __init__(self, lstm_param, lstm_state):
    # store reference to parameters and to activations
    self.state = lstm_state
    self.param = lstm_param
    # non-recurrent input to node
    # 输入序列
    self.x = None
    # non-recurrent input concatenated with recurrent input
    # 输入序列与LSTM在t-1时刻的输出
    self.xc = None

  # 前向计算过程
  def bottom_data_is(self, x, s_prev=None, h_prev=None):
    # if this is the first lstm node in the network
    # 如果该LSTM是网络中隐藏层第一个LSTM节点
    if s_prev == None: s_prev = np.zeros_like(self.state.s)
    if h_prev == None: h_prev = np.zeros_like(self.state.h)
    # save data for use in backprop
    self.s_prev = s_prev
    self.h_prev = h_prev

    # concatenate x(t) and h(t-1)
    # 合并x(t)和h(t-1)
    xc = np.hstack((x, h_prev))
    # 计算LSTM内各状态
    self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
    self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
    self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
    self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
    self.state.s = self.state.g * self.state.i + s_prev * self.state.f
    # self.state.h的计算与经典论文内的计算方法不一样
    self.state.h = self.state.s * self.state.o
    # 经典的算法针对state.s还有一个tanh过程：
    # self.state.h = np.tanh(self.state.s) * self.state.o

    self.x = x
    self.xc = xc

  # 反向梯度计算过程
  # 对于一个LSTM节点来说，需要接受关于h和s两个状态的梯度变化才能逐步回溯其他各门状态的梯度变化
  def top_diff_is(self, top_diff_h, top_diff_s):
    # notice that top_diff_s is carried along the constant error carousel
    ds = self.state.o * top_diff_h + top_diff_s
    do = self.state.s * top_diff_h
    di = self.state.g * ds
    dg = self.state.i * ds
    df = self.s_prev * ds

    # diffs w.r.t. vector inside sigma / tanh function
    di_input = (1. - self.state.i) * self.state.i * di
    df_input = (1. - self.state.f) * self.state.f * df
    do_input = (1. - self.state.o) * self.state.o * do
    dg_input = (1. - self.state.g ** 2) * dg

    # diffs w.r.t. inputs
    self.param.wi_diff += np.outer(di_input, self.xc)
    self.param.wf_diff += np.outer(df_input, self.xc)
    self.param.wo_diff += np.outer(do_input, self.xc)
    self.param.wg_diff += np.outer(dg_input, self.xc)
    self.param.bi_diff += di_input
    self.param.bf_diff += df_input
    self.param.bo_diff += do_input
    self.param.bg_diff += dg_input

    # compute bottom diff
    dxc = np.zeros_like(self.xc)
    dxc += np.dot(self.param.wi.T, di_input)
    dxc += np.dot(self.param.wf.T, df_input)
    dxc += np.dot(self.param.wo.T, do_input)
    dxc += np.dot(self.param.wg.T, dg_input)

    # save bottom diffs
    self.state.bottom_diff_s = ds * self.state.f
    self.state.bottom_diff_x = dxc[:self.param.x_dim]
    self.state.bottom_diff_h = dxc[self.param.x_dim:]


class LstmNetwork():
  def __init__(self, lstm_param):
    self.lstm_param = lstm_param
    self.lstm_node_list = []
    # input sequence
    self.x_list = []

  def y_list_is(self, y_list, loss_layer):
    """
    Updates diffs by setting target sequence
    with corresponding loss layer.
    Will *NOT* update parameters.  To update parameters,
    call self.lstm_param.apply_diff()
    """
    assert len(y_list) == len(self.x_list)
    idx = len(self.x_list) - 1
    # first node only gets diffs from label ...
    loss = loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
    diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
    # here s is not affecting loss due to h(t+1), hence we set equal to zero
    diff_s = np.zeros(self.lstm_param.mem_cell_ct)
    self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
    idx -= 1

    ### ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
    ### we also propagate error along constant error carousel using diff_s
    while idx >= 0:
      loss += loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
      diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
      diff_h += self.lstm_node_list[idx + 1].state.bottom_diff_h
      diff_s = self.lstm_node_list[idx + 1].state.bottom_diff_s
      self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
      idx -= 1

    return loss

  def x_list_clear(self):
    self.x_list = []

  def x_list_add(self, x):
    self.x_list.append(x)
    if len(self.x_list) > len(self.lstm_node_list):
      # need to add new lstm node, create new state mem
      lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
      self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))

    # get index of most recent x input
    idx = len(self.x_list) - 1
    if idx == 0:
      # no recurrent inputs yet
      self.lstm_node_list[idx].bottom_data_is(x)
    else:
      s_prev = self.lstm_node_list[idx - 1].state.s
      h_prev = self.lstm_node_list[idx - 1].state.h
      self.lstm_node_list[idx].bottom_data_is(x, s_prev, h_prev)