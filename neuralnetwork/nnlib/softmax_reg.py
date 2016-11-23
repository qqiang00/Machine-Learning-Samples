'''
  this file includes functions often used in neural network computation
'''
import numpy as np
import random
class softmax_reg():

  @staticmethod
  def init(presize, outsize=1):
    w = np.random.randn(presize+1, outsize)/np.sqrt(presize+1+outsize)
    w[0,:] = 0  # bias
    return w

  @staticmethod
  def forward(in_data,w):

    b,d = in_data.shape
    x = np.zeros((b,d+1))
    x[:,0] = 1 # x with bias
    x[:,1:] = in_data

    assert (x.shape[1] == w.shape[0]), "wrong size of X or W"
    output = np.exp(x.dot(w))
    dev = np.sum(output)  #sum of all elements in matrix output
    output = output * 1. / dev
    cache = {}
    cache["w"] = w
    cache["output"] = output
    cache["x"] = x
    return output, cache

  @staticmethod
  def backward(label_y, cache):
    w = cache["w"]
    output = cache["output"]
    x = cache["x"]
    # shape of label_y : (b,self.outsize)

    assert (label_y.shape[1] == output.shape[1]), "wrong size of delta output"
    dy = output - label_y
    loss = - np.log(np.sum(label_y * output))
    dw = np.dot(x.T, dy)
    dx = np.dot(dy,w.T)[:,1:]

    return loss, dx, dw,

def gradient_check(epsinon=0.001):
  # select a parameter theta

  # prepare one sample to check
  x = np.random.randn(2,100)
  y = np.zeros((2,10))
  y[1,3] = 1

  # build a softmax classifier
  w = softmax_reg.init(100,10)

  # check 100 cases:
  for i in range(100):
    s0 = 0#random.randint(0,w.shape[0]-1)
    s1 = random.randint(0,w.shape[1]-1)

    # compute value1 code
    _,cache = softmax_reg.forward(x,w)
    loss, dx, dw = softmax_reg.backward(y,cache)
    value1= dw[s0,s1]

    # compute gradient of w[s0,s1] from formula:
    # J(theta + epsinon) - J(theta - epsinon) / (2 * epsinon)

    old_theta = w[s0,s1]
    w[s0,s1] = old_theta + epsinon
    _,cache = softmax_reg.forward(x,w)
    loss_plus, _, _ = softmax_reg.backward(y,cache)

    w[s0,s1] = old_theta - epsinon
    _, cache = softmax_reg.forward(x,w)
    loss_minus, _, _ = softmax_reg.backward(y,cache)

    value2 = (loss_plus - loss_minus ) / (2 * epsinon)
    w[s0,s1] = old_theta

    if abs(value1-value2) > 1e-10:
      print("Wrong! gradient check of w[%d,%d] OK!(%f(from code) %f(from formula))" % (s0, s1, value1, value2))
    else:
      print("OK! gradient of w[%d,%d] OK!(%f(from code) %f(from formula))"%(s0,s1,value1,value2))





def train_softmax_reg(X,Y,batch_size = 50,learning_rate = 0.0003, reg_rate = 0.00001, max_iter = 2000000):
  assert (X.shape[0] == Y.shape[0])

  w = softmax_reg.init(X.shape[1], Y.shape[1])

  sample_size = len(X)
  batch_loss = 0
  batch_dw = 0

  iter = 0
  i = 0
  while iter <= max_iter:

    _, cache = softmax_reg.forward(X[i], w)
    loss, dx, dw = softmax_reg.backward(Y[i], cache)
    batch_loss += loss
    batch_dw += dw

    iter += 1
    if iter % batch_size == 0:
      batch_loss /= batch_size
      batch_loss += reg_rate * np.sum(w * w) / 2
      dw /= batch_size
      dw += reg_rate * w
      w -= learning_rate * dw
      if iter % 1000 == 0:
        print("loss:%10.2f  iter:%5d" %(batch_loss,iter))

      batch_loss = 0
      batch_dw = 0

    i += 1
    if i == sample_size - 1:
      i = 0
  print("end of training.")
  return


def test_softmax_reg():
  X = np.random.randn(200,10) / np.sqrt(20)
  Y = np.zeros((200,4))
  for i in range(Y.shape[0]):
    Y[i][random.randint(0,Y.shape[1]-1)] = 1
  train_softmax_reg(X,Y)




if __name__ == "__main__":
  #test_softmaxer()
  gradient_check(1e-5)
