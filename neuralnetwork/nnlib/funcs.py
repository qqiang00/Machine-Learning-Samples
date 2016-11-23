'''
define activation function used in neural network
'''
import numpy as np
import random

# kinds of activation functions and their derivative functions
def Sigmoid(X):
  return 1./(1+np.exp(-1.*X))

def dSigmoid(Y):
  return Y*(1-Y)

def Tanh(X):
  temp = np.exp(X)
  return (temp - 1./temp) / (temp + 1./temp)

def dTanh(Y):
  return 1 - Y * Y

def ReLU(X):
  # return np.maximum(x,0)
  # return (abs(x) + x)/2.
  return X * (X > 0)  # seems to be the fastest

def dReLU(X):
  return 1 * (X>0) # + 0 * (X<=0)

def Linear(X):
  return X

def dLinear(X):
  return 1

# softplus function is like a smooth edition of ReLU function
def Softplus(X):
  return np.log(1+np.exp(X))

# 关于Softplus的求导：Y=Softplus(X)=np.log(1+np.exp(X))
# dY = ... =Sigmoid(X) != Sigmoid(Y)
# dY = 1 - np.exp(-1*Y)
def dSoftplus(Y):
  return 1 - np.exp(-1*Y)

# dictionary to make a connection between function name and function
act_funcs = {}

act_funcs["sigmoid"] = Sigmoid
act_funcs["tanh"] = Tanh
act_funcs["relu"] = ReLU
act_funcs["linear"] = Linear
act_funcs["softplus"] = Softplus
act_funcs["Sigmoid"] = Sigmoid
act_funcs["Tanh"] = Tanh
act_funcs["Relu"] = ReLU
act_funcs["ReLU"] = ReLU
act_funcs["Linear"] = Linear
act_funcs["Softplus"] = Softplus

der_funcs = {}
der_funcs["sigmoid"] = dSigmoid
der_funcs["tanh"] = dTanh
der_funcs["relu"] = dReLU
der_funcs["linear"] = dLinear
der_funcs["softplus"] = dSoftplus
der_funcs["Sigmoid"] = dSigmoid
der_funcs["Tanh"] = dTanh
der_funcs["Relu"] = dReLU
der_funcs["ReLU"] = dReLU
der_funcs["Linear"] = dLinear
der_funcs["Softplus"] = dSoftplus

# with a dictionary, don't need this function any more
def recognize_dfunc(activate_func = None):
  if activate_func == None or activate_func == Sigmoid:
    return dSigmoid;
  if activate_func == Tanh:
    return dTanh
  if activate_func == ReLU:
    return dReLU;
  if activate_func == Linear:
    return dLinear;
  if activate_func == Softplus:
    return dSoftplus;
  else:
    return dSigmoid;

# gradient_check function is used for
def gradient_check(layer_class, func_list, epsinon=0.001):
  # select a parameter theta

  # prepare one sample to check
  x = np.random.randn(2,100)
  #dy = np.zeros((2, 10))
  b, d = 2,10
  # generate randomized output y as a labeled output for gradient check.
  y = np.random.randn(b, d) / np.sqrt(b + d)

  # initialize a neural network layer.
  w = layer_class.init(100,10)
  # we will check different kind of activate function
  #func_list = ["sigmoid","tanh","relu","linear","softplus"]
  # check cases:
  print("Gradient Check of layer:%s" % (layer_class))
  for i in range(len(func_list)):
    # get current activate function name
    func_name=func_list[i]
    # randomized select a weight to check.
    s0 = random.randint(0,w.shape[0]-1)
    s1 = random.randint(0, w.shape[1] - 1)

    # compute gradient by forward function, stored it in value1.
    _,cache = layer_class.forward(x,w,func_name)
    dy = cache["output"] - y
    loss, dx, dw = layer_class.backward(dy,cache)
    value1 = dw[s0,s1]

    # compute gradient of w[s0,s1] by numerical computation approximation
    # which equals to: J(theta + epsinon) - J(theta - epsinon) / (2 * epsinon)

    old_theta = w[s0,s1]    # store old wight of [s0,s1]
    w[s0,s1] = old_theta + epsinon  # add a tiny epsinon

    _,cache = layer_class.forward(x,w,func_name)
    dy = cache["output"] - y
    loss_plus, _, _ = layer_class.backward(dy,cache)  # get loss of new weight

    w[s0,s1] = old_theta - epsinon  # produce another weight
    _, cache = layer_class.forward(x,w,func_name)
    dy = cache["output"] - y
    loss_minus, _, _ = layer_class.backward(dy,cache) # get loss of second weight

    value2 = (loss_plus - loss_minus) / (2 * epsinon) # value2 is the gradient
    w[s0,s1] = old_theta  # get back the old weight.

    if abs(value1-value2) > 1e-5:
      print("  Wrong! using %s: gradient check of w[%d,%d] (%f(from code) %f(from formula))" % (func_name,s0, s1, value1, value2))
    else:
      print("  OK! using %s: gradient of w[%d,%d] (%f(from code) %f(from formula))"%(func_name,s0,s1,value1,value2))

def test():
  print(Sigmoid(np.array([i*0.5-5 for i in range(30)])))


if __name__ == "__main__":
  test()
