#implemented as I read Andrej Karpathy's post on RNNs.
import numpy as np
#import matplotlib.pyplot as plt
from LSTM import *
from neuralnetwork.nnlib.softmax_reg import *

class RNN_LSTM(object):
  def __init__(self, insize, outsize, hidsize, learning_rate):

    self.insize = insize
    self.hidsize = hidsize
    self.outsize = outsize

    self.WLSTM = LSTM.init(input_size=insize, hidden_size=hidsize)
    self.w_softmax = softmax_reg.init(self.hidsize,self.outsize)

    self.learning_rate = learning_rate

  #give the RNN a sequence of inputs and outputs (seq_length long), and use
  #them to adjust the internal state
  #x和y相当于用于网络训练的"块(batch)"数据，其梯度下降使用的是的块数据梯度下降。
  def train(self, x, y):

    # 预处理x,y使得其适配LSTM的参数格式
    n = len(x)
    b = 1
    X = np.zeros((n, b, self.insize))
    Y = np.zeros((n, b, self.outsize))

    for t in range(n):
      X[t, :, x[t]] = 1
      Y[t, :, y[t]] = 1

    h0 = np.zeros((b, self.hidsize))
    c0 = np.zeros((b, self.hidsize))
    # output of whole network
    yhat = np.zeros_like(Y)

    # output of LSTM
    H, _, _, lstm_cache = LSTM.forward(X, self.WLSTM, c0, h0)


    total_dw = np.zeros_like(self.w_softmax)
    dH = np.zeros_like(H)

    total_loss = 0
    for t in reversed(range(n)):
      # 获取t时刻的输出
      yhat[t], softmax_cache = softmax_reg.forward(H[t],self.w_softmax)
      loss, dH[t], dw = softmax_reg.backward(Y[t], softmax_cache)
      total_loss += loss
      total_dw += dw

    _, dWLSTM, _, _ = LSTM.backward(dH, lstm_cache)

    #total_loss /= n
    total_dw /= n
    #dWLSTM /= n
    self.WLSTM -= self.learning_rate * dWLSTM
    self.w_softmax -= self.learning_rate * total_dw
    return total_loss




  #let the RNN generate text
  def sample(self, seed, n):
    ndxs = []
    b = 1
    h0 = np.zeros((b, self.hidsize))
    c0 = np.zeros((b, self.hidsize))
    h, c = h0, c0
    xhat = np.zeros((1,b,self.insize))
    xhat[0,:,seed] = 1  # transform to 1-of-k

    for t in range(n):
      _, cprev, hprev, lstm_cache = LSTM.forward(xhat, self.WLSTM, c, h)
      p, softmax_cache = softmax_reg.forward(hprev, self.w_softmax)
      # select a int number according to their probabilities stored in variable p.
      ndx = np.random.choice(range(self.insize), p=p.ravel())

      xhat = np.zeros((1,b,self.insize))
      xhat[0,:,ndx] = 1
      c = cprev
      h = hprev

      ndxs.append(ndx)

    return ndxs


def test():# test a very very long txt file.
  # open a text file
  data = open("..//DataSet//rnn_test.txt", 'r').read()  # should be simple plain text file
  # set()查找data里所有不重复的字符
  chars = list(set(data))
  data_size, vocab_size = len(data), len(chars)
  print("data has %d characters, %d unique." % (data_size, vocab_size))

  # make some dictionaries for encoding and decoding from 1-of-k
  # 双向映射的字典，例如：char_to_ix["c"]=0, 则ix_to_char[0]="c"

  char_to_ix = {ch: i for i, ch in enumerate(chars)}
  ix_to_char = {i: ch for i, ch in enumerate(chars)}

  # insize and outsize are len(chars). hidsize is 100. seq_length is 25. learning_rate is 0.1.
  rnn = RNN_LSTM(len(chars), len(chars), 100, 0.1)

  # iterate over batches of input and target output
  seq_length = 20
  losses = []
  # 简便的计算seq_length次smooth_loss
  smooth_loss = -np.log(1.0 / len(chars)) * seq_length  # loss at iteration 0
  losses.append(smooth_loss)

  for i in range(int(len(data) / seq_length)):
    x = [char_to_ix[c] for c in data[i * seq_length:(i + 1) * seq_length]]  # inputs to the RNN
    y = [char_to_ix[c] for c in
         data[i * seq_length + 1:(i + 1) * seq_length + 1]]  # the targets it should be outputting

    if i % 1000 == 0:
      sample_ix = rnn.sample(x[0], 200)
      txt = ''.join([ix_to_char[n] for n in sample_ix])
      print("\n")
      print(txt)
      pass

    loss = rnn.train(x, y)

    # smooth_loss = smooth_loss*0.999 + loss*0.001
    smooth_loss = loss

    if i % 1000 == 0:
      print("iteration %d, smooth_loss = %f" % (i, smooth_loss))
    losses.append(smooth_loss)

  plt.plot(range(len(losses)), losses, 'b', label='smooth loss')
  plt.xlabel('time in thousands of iterations')
  plt.ylabel('loss')
  plt.legend()
  plt.show()


def test1(): # test a short txt file.
  # open a text file
  data = open('..//DataSet//fisherman.txt', 'r').read()  # should be simple plain text file
  chars = list(set(data))
  data_size, vocab_size = len(data), len(chars)
  print("data has %d characters, %d unique." % (data_size, vocab_size))

  # make some dictionaries for encoding and decoding from 1-of-k
  char_to_ix = {ch: i for i, ch in enumerate(chars)}
  ix_to_char = {i: ch for i, ch in enumerate(chars)}

  # insize and outsize are len(chars). hidsize is 100. seq_length is 25. learning_rate is 0.1.
  rnn = RNN_LSTM(len(chars), len(chars), 100, 0.1)

  # iterate over batches of input and target output
  seq_length = 20
  losses = []
  smooth_loss = -np.log(1.0 / len(chars)) * seq_length  # loss at iteration 0
  losses.append(smooth_loss)

  i = 0
  count = 0
  while count < 1e8:
    # for i in range(int(len(data)/seq_length)):
    x = [char_to_ix[c] for c in data[i * seq_length:(i + 1) * seq_length]]  # inputs to the RNN
    y = [char_to_ix[c] for c in
         data[i * seq_length + 1:(i + 1) * seq_length + 1]]  # the targets it should be outputting
    if i % (int(len(data) / seq_length) - 1) == 0:
      i = 0

    if count % 1000 == 0:
      sample_ix = rnn.sample(x[0], 100)
      txt = ''.join([ix_to_char[n] for n in sample_ix])
      print("\n")
      print(txt)

    loss = rnn.train(x, y)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if count % 1000 == 0:
      print("iteration %d, smooth_loss = %f" % (count, smooth_loss))
      losses.append(smooth_loss)
    i += 1
    count += 1

  plt.plot(range(len(losses)), losses, 'b', label='smooth loss')
  plt.xlabel('time in thousands of iterations')
  plt.ylabel('loss')
  plt.legend()
  plt.show()

if __name__ == "__main__":
  test1()