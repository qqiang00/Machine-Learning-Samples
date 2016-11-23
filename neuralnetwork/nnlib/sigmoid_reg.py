'''
  a soigmoid regression output layer
'''
import numpy as np
import random
from nnlib.funcs import *
import nnlib.funcs
from nnlib.nnlayer import nnlayer

# sigmoid_reg class is very like the nnlayer where the only difference is the
# activate function is supposed to choose "sigmoid" function
class sigmoid_reg(nnlayer):

  @staticmethod
  def init(presize=1, outsize=1):
    return nnlayer.init(presize, outsize)

  @staticmethod
  def forward(in_data,w,userless_parameter):

    return nnlayer.forward(in_data, w, func_name = "sigmoid")
    return output, cache

  @staticmethod
  def backward(dy, cache):
    return nnlayer.backward(dy, cache)


if __name__ == "__main__":
  func_list = ["sigmoid", "tanh", "relu", "linear", "softplus"]
  funcs.gradient_check(sigmoid_reg,func_list,1e-5)
