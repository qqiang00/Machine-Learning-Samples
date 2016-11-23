"""
This is a batched LSTM forward and backward pass
the comment is writen by karpathy, except the comment start with #sooda:
#sooda: will add some comment corresponding the equtions (ref: lstm.png)
downloaded from: https://gist.github.com/karpathy/587454dc0146a6ae21fc
"""
import numpy as np
#import code


class LSTM:
  @staticmethod
  def init(input_size, hidden_size, fancy_forget_bias_init=3):
    """
        Initialize parameters of the LSTM (both weights and biases in one matrix)
        One might way to have a positive fancy_forget_bias_init number
        (e.g. maybe even up to 5, in some papers)
        """
    # +1 for the biases, which will be the first row of WLSTM
    # 存放权重和偏置项的矩阵，结构如下
    #    input    forget output  cell state
    #   +-------|-------|-------|-------+
    #   |       |       |       |       | Row0:bias
    #   +-------+-------+-------+-------+
    #   |       |       |       |       |
    #   |       |       |       |       | input size
    #   |       |       |       |       |
    #   |       |       |       |       |
    #   +-------+-------+-------+-------+
    #   |       |       |       |       |
    #   |       |       |       |       | hidden size
    #   |       |       |       |       |
    #   +-------+-------+-------+-------+
    WLSTM = np.random.randn(input_size + hidden_size + 1, 4 * hidden_size) \
            / np.sqrt(input_size + hidden_size)
    WLSTM[0, :] = 0  # initialize biases to zero
    if fancy_forget_bias_init != 0:
      # forget gates get little bit negative bias initially to encourage them to be turned off
      # remember that due to Xavier initialization above, the raw output activations from gates before
      # nonlinearity are zero mean and on order of standard deviation ~1
      WLSTM[0, hidden_size:2 * hidden_size] = fancy_forget_bias_init
    return WLSTM

  @staticmethod
  def forward(X, WLSTM, c0=None, h0=None):
    """
        X should be of shape (n,b,input_size), where n = length of sequence, b = batch size
        #sooda: WLSTM is the weight of the total nets.
        #sooda: rows is input_size + hidden_size + basis. cols is input, forget, output, cell gate weights.
        #sooda: IFOG: o-d input, d-2d forget, 2d-3d output, 3d-end cell gate.
        #sooda: every one is in the same size with hidden
        #sooda: for equations 1-4, the items are all x_t, h_t\-1
        #sooda: equations 1-4 can vectorize, and parallelize
        """
    n, b, input_size = X.shape  # n 输入序列的长度; b batch size; input size:LSTM接受输入的维度
    d = int(WLSTM.shape[1] / 4)  # hidden size （LSTM细胞规模，细胞状态数量）
    if c0 is None: c0 = np.zeros((b, d))  # 前一时刻细胞内的状态
    if h0 is None: h0 = np.zeros((b, d))  # 前一时刻LSTM的输出

    # Perform the LSTM forward pass with X as the input
    xphpb = WLSTM.shape[0]  # x plus h plus bias, lol; 网络参数矩阵的行数，其数值相当于1+输入维度+细胞状态数
    Hin = np.zeros((n, b, xphpb))  # input [1, xt, ht-1] to each tick of the LSTM
    Hout = np.zeros((n, b, d))  # hidden representation of the LSTM (gated cell content)
    IFOG = np.zeros((n, b, d * 4))  # input, forget, output, gate (IFOG) #sooda:this is the value before activation
    IFOGf = np.zeros((n, b, d * 4))  # after nonlinearity #sooda: value of activation
    C = np.zeros((n, b, d))  # cell content
    Ct = np.zeros((n, b, d))  # tanh of cell content
    for t in range(n):  # 对于序列内的每一个step，相当于每一个t时刻
      # concat [x,h] as input to the LSTM
      prevh = Hout[t - 1] if t > 0 else h0  #设置t-1时刻的h作为t时刻输入的一部分
      Hin[t, :, 0] = 1  # bias  # 任一t时刻输入的偏置项都设定为1
      Hin[t, :, 1:input_size + 1] = X[t]
      Hin[t, :, input_size + 1:] = prevh  # 至此已设置好完整的t时刻的输入
      # compute all gate activations. dots: (most work is this line)
      IFOG[t] = Hin[t].dot(WLSTM) # 先所有点乘一下
      # non-linearities # 除存放细胞状态的矩阵外都经过sigmoid变换
      IFOGf[t, :, :3 * d] = 1.0 / (1.0 + np.exp(-IFOG[t, :, :3 * d]))  # sigmoids; these are the gates

      # 存放细胞状态的矩阵部分使用tanh变换
      IFOGf[t, :, 3 * d:] = np.tanh(IFOG[t, :, 3 * d:])  # tanh  #sooda: equation 1-4 done

      # compute the cell activation 开始计算激活值
      prevc = C[t - 1] if t > 0 else c0   # 设置t-1时刻的细胞状态，C0是传递进来的参数

      C[t] = IFOGf[t, :, :d] * IFOGf[t, :, 3 * d:] + IFOGf[t, :, d:2 * d] * prevc
      Ct[t] = np.tanh(C[t])  # sooda: equation 5
      Hout[t] = IFOGf[t, :, 2 * d:3 * d] * Ct[t]  # sooda: equation6

    cache = {}
    cache['WLSTM'] = WLSTM
    cache['Hout'] = Hout
    cache['IFOGf'] = IFOGf
    cache['IFOG'] = IFOG
    cache['C'] = C
    cache['Ct'] = Ct
    cache['Hin'] = Hin
    cache['c0'] = c0
    cache['h0'] = h0

    # return C[t], as well so we can continue LSTM with prev state init if needed
    return Hout, C[t], Hout[t], cache

  @staticmethod
  def backward(dHout_in, cache, dcn=None, dhn=None):

    WLSTM = cache['WLSTM']
    Hout = cache['Hout']
    IFOGf = cache['IFOGf']
    IFOG = cache['IFOG']
    C = cache['C']
    Ct = cache['Ct']
    Hin = cache['Hin']
    c0 = cache['c0']
    h0 = cache['h0']
    n, b, d = Hout.shape
    input_size = WLSTM.shape[0] - d - 1  # -1 due to bias

    # backprop the LSTM
    dIFOG = np.zeros(IFOG.shape)
    dIFOGf = np.zeros(IFOGf.shape)
    dWLSTM = np.zeros(WLSTM.shape)
    dHin = np.zeros(Hin.shape)
    dC = np.zeros(C.shape)
    dX = np.zeros((n, b, input_size))
    dh0 = np.zeros((b, d))
    dc0 = np.zeros((b, d))

    # dHout_in是传进来的参数，是整体的loss
    dHout = dHout_in.copy()  # make a copy so we don't have any funny side effects
    if dcn is not None: dC[n - 1] += dcn.copy()  # carry over gradients from later
    if dhn is not None: dHout[n - 1] += dhn.copy()  # sooda: dHout is output loss
    for t in reversed(range(n)):
      tanhCt = Ct[t]
      dIFOGf[t, :, 2 * d:3 * d] = tanhCt * dHout[t]  # sooda: partial derivative of Output, eq6
      # backprop tanh non-linearity first then continue backprop
      dC[t] += (1 - tanhCt ** 2) * (IFOGf[t, :, 2 * d:3 * d] * dHout[t])  # sooda: partial derivative of Cell, eq7

      if t > 0:
        dIFOGf[t, :, d:2 * d] = C[t - 1] * dC[t]  # sooda: partial derivative of Forget, eq5
        dC[t - 1] += IFOGf[t, :, d:2 * d] * dC[t]  # sooda: partial derivative of Cell t-1, eq5
      else:
        dIFOGf[t, :, d:2 * d] = c0 * dC[t]
        dc0 = IFOGf[t, :, d:2 * d] * dC[t]
      dIFOGf[t, :, :d] = IFOGf[t, :, 3 * d:] * dC[t]  # sooda: partial derivative of Cell, eq5
      dIFOGf[t, :, 3 * d:] = IFOGf[t, :, :d] * dC[t]  # sooda: partial derivative of Input, eq5
      # qiang: 上两行sooda的注释好像弄反了，第一行是Input，第二行是Cell

      # backprop activation functions
      dIFOG[t, :, 3 * d:] = (1 - IFOGf[t, :, 3 * d:] ** 2) * dIFOGf[t, :,
                                                             3 * d:]  # sooda: sigmoid activation derivative
      y = IFOGf[t, :, :3 * d]
      dIFOG[t, :, :3 * d] = (y * (1.0 - y)) * dIFOGf[t, :, :3 * d]  # sooda: tanh activation derivative
      # qiang: 上两个注释好像也弄反了，前一个是tanh，后一个是sigmoid

      # backprop matrix multiply
      dWLSTM += np.dot(Hin[t].transpose(), dIFOG[t])  # sooda: eq1-4 derivative. as the derivative of f(x) = Wx
      dHin[t] = dIFOG[t].dot(WLSTM.transpose())

      # backprop the identity transforms into Hin
      dX[t] = dHin[t, :, 1:input_size + 1]
      if t > 0:
        dHout[t - 1, :] += dHin[t, :, input_size + 1:]
      else:
        dh0 += dHin[t, :, input_size + 1:]

    return dX, dWLSTM, dc0, dh0


# -------------------
# TEST CASES
# -------------------



def checkSequentialMatchesBatch():
  """ check LSTM I/O forward/backward interactions """

  n, b, d = (5, 3, 4)  # sequence length, batch size, hidden size
  input_size = 10
  WLSTM = LSTM.init(input_size, d)  # input size, hidden size
  X = np.random.randn(n, b, input_size)
  h0 = np.random.randn(b, d)
  c0 = np.random.randn(b, d)

  # sequential forward
  cprev = c0
  hprev = h0
  caches = [{} for t in range(n)]
  Hcat = np.zeros((n, b, d))
  for t in range(n):
    xt = X[t:t + 1]
    _, cprev, hprev, cache = LSTM.forward(xt, WLSTM, cprev, hprev)
    caches[t] = cache
    Hcat[t] = hprev

  # sanity check: perform batch forward to check that we get the same thing
  H, _, _, batch_cache = LSTM.forward(X, WLSTM, c0, h0)
  assert np.allclose(H, Hcat), 'Sequential and Batch forward don''t match!'

  # eval loss
  wrand = np.random.randn(*Hcat.shape)
  loss = np.sum(Hcat * wrand)
  dH = wrand

  # get the batched version gradients
  BdX, BdWLSTM, Bdc0, Bdh0 = LSTM.backward(dH, batch_cache)

  # now perform sequential backward
  dX = np.zeros_like(X)
  dWLSTM = np.zeros_like(WLSTM)
  dc0 = np.zeros_like(c0)
  dh0 = np.zeros_like(h0)
  dcnext = None
  dhnext = None
  for t in reversed(range(n)):
    dht = dH[t].reshape(1, b, d)
    dx, dWLSTMt, dcprev, dhprev = LSTM.backward(dht, caches[t], dcnext, dhnext)
    dhnext = dhprev
    dcnext = dcprev

    dWLSTM += dWLSTMt  # accumulate LSTM gradient
    dX[t] = dx[0]
    if t == 0:
      dc0 = dcprev
      dh0 = dhprev

  # and make sure the gradients match
  print("Making sure batched version agrees with sequential version: (should all be True)")
  print(np.allclose(BdX, dX))
  print(np.allclose(BdWLSTM, dWLSTM))
  print(np.allclose(Bdc0, dc0))
  print(np.allclose(Bdh0, dh0))


def checkBatchGradient():
  """ check that the batch gradient is correct """

  # lets gradient check this beast
  n, b, d = (5, 3, 4)  # sequence length, batch size, hidden size
  input_size = 10
  WLSTM = LSTM.init(input_size, d)  # input size, hidden size
  X = np.random.randn(n, b, input_size)
  h0 = np.random.randn(b, d)
  c0 = np.random.randn(b, d)

  # batch forward backward
  H, Ct, Ht, cache = LSTM.forward(X, WLSTM, c0, h0)
  wrand = np.random.randn(*H.shape)
  loss = np.sum(H * wrand)  # weighted sum is a nice hash to use I think
  dH = wrand
  dX, dWLSTM, dc0, dh0 = LSTM.backward(dH, cache)

  def fwd():
    h, _, _, _ = LSTM.forward(X, WLSTM, c0, h0)
    return np.sum(h * wrand)

  # now gradient check all
  delta = 1e-5
  rel_error_thr_warning = 1e-2
  rel_error_thr_error = 1
  tocheck = [X, WLSTM, c0, h0]
  grads_analytic = [dX, dWLSTM, dc0, dh0]
  names = ['X', 'WLSTM', 'c0', 'h0']
  for j in range(len(tocheck)):
    mat = tocheck[j]
    dmat = grads_analytic[j]
    name = names[j]
    # gradcheck
    for i in range(mat.size):
      old_val = mat.flat[i]
      mat.flat[i] = old_val + delta
      loss0 = fwd()
      mat.flat[i] = old_val - delta
      loss1 = fwd()
      mat.flat[i] = old_val

      grad_analytic = dmat.flat[i]
      grad_numerical = (loss0 - loss1) / (2 * delta)

      if grad_numerical == 0 and grad_analytic == 0:
        rel_error = 0  # both are zero, OK.
        status = 'OK'
      elif abs(grad_numerical) < 1e-7 and abs(grad_analytic) < 1e-7:
        rel_error = 0  # not enough precision to check this
        status = 'VAL SMALL WARNING'
      else:
        rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
        status = 'OK'
        if rel_error > rel_error_thr_warning: status = 'WARNING'
        if rel_error > rel_error_thr_error: status = '!!!!! NOTOK'

      # print stats
      print('%s checking param %s index %s (val = %+8f), analytic = %+8f, numerical = %+8f, relative error = %+8f' \
      % (status, name, np.unravel_index(i, mat.shape), old_val, grad_analytic, grad_numerical, rel_error))



if __name__ == "__main__":
  checkSequentialMatchesBatch()
  input('check OK, press key to continue to gradient check')
  checkBatchGradient()
  print("every line should start with OK. Have a nice day!")