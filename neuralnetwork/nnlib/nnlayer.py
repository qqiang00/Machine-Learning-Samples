'''
  a soigmoid regression output layer
'''
import numpy as np
import random
from .funcs import *
# 定义一个神经网络层
class nnlayer():

  @staticmethod
  def init(presize=1, outsize=1):
    weight = np.random.randn(presize+1, outsize)/np.sqrt(presize+1+outsize)
    weight[0,:] = 0  # bias
    return weight

  # 前向传播函数，func_name定义了采用的激活函数
  @staticmethod
  def forward(in_data,w,func_name = "Sigmoid"):

    b,d = in_data.shape
    x = np.zeros((b,d+1))
    x[:,0] = 1 # x with bias
    x[:,1:] = in_data

    assert (x.shape[1] == w.shape[0]), "wrong size of X or W"

    # act_funcs is a dictionary allowing us to declare a function with a string
    # act_funcs 在func.py内定义
    output = act_funcs[func_name](x.dot(w))
    # 声明一个字典，存放相关数据，给反向传播函数
    cache = {}
    cache["w"] = w
    cache["output"] = output
    cache["x"] = x
    cache["act_func"] = func_name #反向传播函数需要知道激活函数类型
    return output, cache

  @staticmethod
  def backward(dy, cache):
    # 反向传播函数的参数dy和该函数最后返回值其一dx地位相同，也就是说，
    # 对于更前一个网络层，
    # 其反向传播函数接受的dy参数就是其后一层反向传播函数的返回值dx
    w = cache["w"]
    output = cache["output"]
    x = cache["x"]
    func_name = cache["act_func"]
    # der_funcs is a dictionary stored derivative functions with index of their names
    # der_funcs 也是一个字典，在funcs里定义，根据激活函数名称返回对应的求导函数
    # 求导函数的参数是输出值而不是输入值
    deriv_func = der_funcs[func_name]

    # shape of label_y : (b,self.outsize)

    assert (dy.shape[1] == output.shape[1]), "wrong size of delta output"
    # 对于大多数网络层，loss我们都采用的是差的平方和函数
    # 选用什么样的loss计算方式对于计算后续的delta(dy)以及梯度检验是不一样的
    loss = np.sum(dy*dy)/2    # here dy is just the difference of output and labeld_y
    #loss = - np.log(np.sum((output - dy) * output)+0.0001) if for softmax regression
    # here is the delta y accompanied with square sum loss function
    dy = dy * deriv_func(output)
    # compute delta weight and delta x which is also the delta y for previous layer.
    dw = np.dot(x.T, dy)
    dx = np.dot(dy,w.T)[:,1:]

    return loss, dx, dw,



if __name__ == "__main__":
  func_list = ["sigmoid", "tanh", "relu", "linear", "softplus"]
  gradient_check(nnlayer,func_list,1e-5)
