"""
本段代码使用了多种方法，来运算描述异或问题，适用与初学NN者了解：矩阵运算，神经元工作机制，神经网络偏置项与权重统一考虑运算
作者：叶强 qqiangye@gmail.com
"""

import numpy as np

# 编程方法1：
# 输入层到中间层的权重结构
# 构造一个2行×2列的矩阵，矩阵所有元素值为0
w_ih = np.zeros((2, 2))
# 单个指定为矩阵的元素值
w_ih[0, 0] = 3
w_ih[0, 1] = -3
w_ih[1, 0] = -3
w_ih[1, 1] = 3
# 显示这个矩阵
#print("weight in to hide:")
#print(w_ih)

# 中间层到输出层的权重
# 构造一个结构为2行×1列，所有元素值均为2的矩阵
w_ho = np.ones((2, 1))*2
#print("weight hide to out")
#print(w_ho)

# 中间层神经元的偏置项矩阵：2行×1列
b_h = np.zeros((2, 1))
# 该矩阵所有元素赋值为-2
b_h[:, :] = -2
# 上2句也可以合并为：
# b_h = np.ones((2,1))*(-2)
#print("bias of hide")
#print(b_h)

# 输出神经元的偏置项，单个变量，不使用矩阵
b_o = 0
#print("bias of out")
#print(b_o)

# f_z函数,建立z到激活值的映射，非矩阵函数
def f_z(z):
  return (1 if z>0 else 0)
  

# 网络的预测运算
def predict(x1,x2):
  z_h0=x1*w_ih[0, 0]+x2*w_ih[1, 0]+b_h[0, 0]
  a_h0=f_z(z_h0)  # 第一个中间层神经元激活值
  z_h1=x1*w_ih[0, 1]+x2*w_ih[1, 1]+b_h[1, 0]
  a_h1=f_z(z_h1)  # 第二个中间层神经元激活值
  
  z_o = a_h0 * w_ho[0, 0] + a_h1 * w_ho[1, 0] + b_o
  output = f_z(z_o) # 输出层神经元激活值，也就是整个网络的输出
  return output

print("Method1: step by step")
print("XOR (0,0) = %d" %(predict(0, 0)))
print("XOR (0,1) = %d" %(predict(0, 1)))
print("XOR (1,0) = %d" %(predict(1, 0)))
print("XOR (1,1) = %d" %(predict(1, 1)))

# --------------------------------------- #
# 编程方法二：矢量化（矩阵）编程
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
# print(data)
# 矩阵函数
def f2_z(z):
  return 1*(z>0)

def predict2(data):
  n = len(data)
  X = data[:, 0:2]
 
  for i in range(n):
    x = X[i].reshape(1,2)
    a_h = f2_z(np.dot(x,w_ih) + b_h.reshape(1,2))
    output = f2_z(np.dot(a_h, w_ho) + b_o)
    print("XOR of %s = %s" % (x, output))
  pass

print("Method 2: Matrix")
predict2(data)

# ------------------------------------------- #
# method3: bias as an element of weight matrix
w_ih = [[-2,-2],[3,-3],[-3,3]]
w_ih = np.array(w_ih)
w_ho = [-1,2,2]
w_ho = np.array(w_ho).reshape(-1,1)

def predict3(data):
  n = len(data)
  X = data[:, 0:2]
  for i in range(n):
    x = np.ones((1,3))
    a_h = np.ones((1,3))
    x[:,1:] = X[i].reshape(1,2) # 3*1
    
    h = f2_z(np.dot(x, w_ih)) # 1*3 dot 3*2 = 1*2
    a_h[:,1:] = h.reshape(1,2)  # 3*1
    
    output = f2_z(np.dot(a_h, w_ho)) # 1*3 dot 3*1 = 1*1
    # x now is a matrix of 3*1, remove the element to bias.
    print("XOR of %s = %s" % (x[:,1:], output))
  pass

print("Method3: bias as an row of weight matrix")
predict3(data)
