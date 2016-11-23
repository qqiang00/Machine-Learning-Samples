import numpy as np
from nnlib.nnlayer import *

# 初始化训练数据集
data = np.zeros((4,3))
data[1,1],data[1,2]=1,1
data[2,0],data[2,2]=1,1
data[3,0],data[3,1]=1,1
# data =
#   x1  x2  y
#[[ 0.  0.  0.]
# [ 0.  1.  1.]
# [ 1.  0.  1.]
# [ 1.  1.  0.]]

# 初始化网络参数：3层之间的2个权重矩阵，程序自动添加偏置项
w_in_to_hide = nnlayer.init(2,2)
w_hide_to_out = nnlayer.init(2,1)

# 设置网络学习参数：学习速率，最大迭代次数，最大可接受Loss
# 设置权重惩罚项 lamda
learn_rate = 1
lamda = 0
total_loss = 0
acceptable_loss = 1e-6
maximum_iter = 500000

for ite in range(maximum_iter):
  j = ite % 4
  in_data = data[j].reshape(1,-1)
  X = in_data[:,0:2]
  Y = in_data[:,2]
  hide, cache_hide = nnlayer.forward(X,w_in_to_hide,"sigmoid")
  output, cache_out = nnlayer.forward(hide,w_hide_to_out,"sigmoid")
  dy = output - Y
  loss, dhide,dw_hide_to_out = nnlayer.backward(dy,cache_out)
  _, _, dw_in_to_hide = nnlayer.backward(dhide,cache_hide)
  # update weight (bias included)
  w_hide_to_out -= learn_rate * dw_hide_to_out
  w_in_to_hide -= learn_rate * dw_in_to_hide

  # update weight with weight regulation
  # it seems it doesn't work for XOR problems.
  #loss += 0.5 * lamda * np.sum((np.square(w_in_to_hide[1:,:]) +\
  #                              np.square(w_hide_to_out[1:,:])))
  
  #w_hide_to_out[1:,:] -= learn_rate * dw_hide_to_out[1:,:] +\
  #                       lamda * w_hide_to_out[1:,:]
  #w_hide_to_out[0:1,:] -= learn_rate * dw_hide_to_out[0:1,:]
  #
  #w_in_to_hide[1:,:] -= learn_rate * dw_in_to_hide[1:,:] +\
  #                      lamda * w_in_to_hide[1:,:]
  #w_in_to_hide[0:1,:] -= learn_rate * dw_in_to_hide[0:1,:]

  if ite % 10000 == 0:
    print("Iter:%7d Loss:%f"%(ite,loss))
  if loss <= acceptable_loss:
    break

print("weight hide_to_out:")
print(w_hide_to_out)
print("weight in_to_hide")
print(w_in_to_hide)
print("Prediction:")
for i in range(4):
  in_data = data[i].reshape(1,-1)
  X = in_data[:,0:2]
  Y = in_data[:,2]
  hide, _ = nnlayer.forward(X,w_in_to_hide,"sigmoid")
  output, _ = nnlayer.forward(hide,w_hide_to_out,"sigmoid")
  print("   X=%s;Y=%s;predict:%s"%(X,Y,output))
