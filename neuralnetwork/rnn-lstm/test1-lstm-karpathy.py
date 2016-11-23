from LSTM import *
import numpy as np
import matplotlib.pyplot as plt


# let the RNN generate text
def sample(WLSTM, seed, n):

  b = 1
  ndxs = []

  d = int(WLSTM.shape[1] / 4)
  input_size = WLSTM.shape[0] - 1 - d
  x0 = np.zeros((1, b, input_size))
  x0[0,:,seed] = 1

  h0= np.random.randn(b, d)
  c0 = np.random.randn(b, d)
  cprev = c0
  hprev = h0
  x = x0
  for t in range(n):

    _, cprev, hprev, _ = LSTM.forward(x, WLSTM, cprev, hprev)

    y = hprev

    p = np.exp(y) / np.sum(np.exp(y))
    # select a int number according to their probabilities stored in variable p.
    ndx = np.random.choice(range(input_size), p=p.ravel())

    y = np.zeros_like(hprev)
    y[0, :, ndx] = 1
    x = y

    ndxs.append(ndx)
  return ndxs



def test():  # test a very very long txt file.
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

  # 将索引数字转换为二进制数字，减少网络
  # bit_size = np.math.ceil(np.math.log(vocab_size,2))

  # insize and outsize are len(chars). hidsize is 100. seq_length is 25. learning_rate is 0.1.
  WLSTM = LSTM.init(vocab_size,100)

  #lstm_rnn = LSTM.init(bit_size,100)

  # iterate over batches of input and target output
  seq_length = 25
  losses = []
  # 简便的计算seq_length次smooth_loss
  smooth_loss = -np.log(1.0 / len(chars)) * seq_length  # loss at iteration 0
  losses.append(smooth_loss)

  for i in range(int(len(data) / seq_length)):
    x = [char_to_ix[c] for c in data[i * seq_length:(i + 1) * seq_length]]  # inputs to the RNN
    y = [char_to_ix[c] for c in
         data[i * seq_length + 1:(i + 1) * seq_length + 1]]  # the targets it should be outputting

    if i % 1000 == 0:
      sample_ix = sample(WLSTM, x[0], 200)
      txt = ''.join([ix_to_char[n] for n in sample_ix])
      print("\n")
      print(txt)

    n = len(x)
    b = 1

    X = np.random.randn(n, b, vocab_size)

    for t in range(n):
      X[t,:,x[t]] = 1

    h0 = np.random.randn(b, d)
    c0 = np.random.randn(b, d)


    # batch forward backward
    H, Ct, Ht, cache = LSTM.forward(X, WLSTM, c0, h0)
    loss = 0
    Y = np.zeros_like(H)
    for t in reversed(range(n)):
      Y[t,:,y[t]] = 1
      loss += -np.log(H[t,:,y[t]])

    #Y = np.random.randn(*H.shape)
    #loss = np.sum(H * Y)  # weighted sum is a nice hash to use I think
    dH = H - Y

    dX, dWLSTM, dc0, dh0 = LSTM.backward(dH, cache)

    WLSTM += dWLSTM


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

if __name__ == "__main__":
  test()
  input('check OK, press key to continue to gradient check')

  print("every line should start with OK. Have a nice day!")