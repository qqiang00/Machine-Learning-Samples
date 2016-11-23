#implemented as I read Andrej Karpathy's post on RNNs.
import numpy as np
import matplotlib.pyplot as plt

class RNN(object):

    def __init__(self, insize, outsize, hidsize, learning_rate):
        self.insize = insize

        self.h = np.zeros((hidsize , 1))#a [h x 1] hidden state stored from last batch of inputs

        #parameters
        self.W_hh = np.random.randn(hidsize, hidsize)*0.01#[h x h]
        self.W_xh = np.random.randn(hidsize, insize)*0.01#[h x x]
        self.W_hy = np.random.randn(outsize, hidsize)*0.01#[y x h]
        self.b_h = np.zeros((hidsize, 1))#biases
        self.b_y = np.zeros((outsize, 1))

        #the Adagrad gradient update relies upon having a memory of the sum of squares of dparams
        #不太明白ada开头的这些参数的意义，从过程看是d开头的参数的平方和，在
        #更新参数的时候，这些平方和作为分母来降低更新的速率
        self.adaW_hh = np.zeros((hidsize, hidsize))
        self.adaW_xh = np.zeros((hidsize, insize))
        self.adaW_hy = np.zeros((outsize, hidsize))
        self.adab_h = np.zeros((hidsize, 1))
        self.adab_y = np.zeros((outsize, 1))

        self.learning_rate = learning_rate

    #give the RNN a sequence of inputs and outputs (seq_length long), and use
    #them to adjust the internal state
    #x和y相当于用于网络训练的"块(batch)"数据，其梯度下降使用的是折中的块数据梯度下降。
    def train(self, x, y):
        #=====initialize=====
        #{}赋值的变量都是字典变量，相当于把batch—size内的所有数据都缓存起来，
        #用于计算参数的梯度变化
        xhat = {}#holds 1-of-k representations of x
        yhat = {}#holds 1-of-k representations of predicted y (unnormalized log probs)
        p = {}#the normalized probabilities of each output through time
        h = {}#holds state vectors through time
        #[-1]中的-1不是链表或数组的索引，而是字典的键值，使用-1也恰好是因为键值基本上
        #是从0开始递增的整数序列
        h[-1] = np.copy(self.h)#we will need to access the previous state to calculate the current state
        #zero_like只是使用参数矩阵的shape构建一个值全为0的矩阵
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        dh_next = np.zeros_like(self.h)

        #=====forward pass=====
        loss = 0
        for t in range(len(x)):
            #以下两行是构建RNN网络的输入，先全赋值为0
            xhat[t] = np.zeros((self.insize, 1))
            #随后，将x[t]所代表的字符（值为整数）作为索引设定xhat[t]矩阵相应位置的值为1
            #请注意xhat是一个字典，而xhat[t]才是一个输入矩阵，其shape为（insize，1）
            xhat[t][x[t]] = 1#xhat[t] = 1-of-k representation of x[t]
            #计算中间层节点的输出：其有2个来源，一个来自输入层，另一个来自前一时刻的中间层输出
            #这里使用到了h[-1],需提前赋值
            #tanh的激活比sigmoid的激活方法效果要好
            h[t] = np.tanh(np.dot(self.W_xh, xhat[t]) + np.dot(self.W_hh, h[t-1]) + self.b_h)#find new hidden state
            #yhat[t]并不是激活值，而只是节点输入的权重和与偏置量的和
            yhat[t] = np.dot(self.W_hy, h[t]) + self.b_y#find unnormalized log probabilities for next chars
            #计算各个输出节点相对概率值，其值得和加起来为1.
            p[t] = np.exp(yhat[t]) / np.sum(np.exp(yhat[t]))#find probabilities for next chars
            #计算损失值，采用交叉熵的计算方法。其中[y[t],0]是字典p下键为t时的输出概率矩阵的索引
            #该索引也可以表示为[y[t]][0].由于矩阵的shape是（-1，1），因此第二维的参数只能是0
            #y[t]表示的是训练集中的输出字符的位置。如果网络的输出在该位置的概率也是1的话，此时
            #loss则为0，如果网络输出是p(<1)，则交叉熵为-log(p),p越小，log(p)越接近负无穷大
            #-log(p)则越大，表示误差（损失）越大。
            loss += -np.log(p[t][y[t],0])#softmax (cross-entropy loss)

        #=====backward pass: compute gradients going backwards=====
        for t in reversed(range(len(x))):
            #backprop into y. see http://cs231n.github.io/neural-networks-case-study/
            # #grad if confused here
            # 比较巧妙的计算dy的形式，在训练数据集的输出中，只有一个节点是1，其余均为0
            # 我们仅需要先把网络输出赋值给dy，然后把参考输出为1的那个减去即可。
            dy = np.copy(p[t])
            dy[y[t]] -= 1

            #find updates for y
            # 根据输出层残差及连接权重计算中间层到输出层的权重和偏置量变化
            dW_hy += np.dot(dy, h[t].T)
            db_y += dy

            #backprop into h and through tanh nonlinearity
            # 计算中间层的残差，有些地方不是很明白
            dh = np.dot(self.W_hy.T, dy) + dh_next
            dh_raw = (1 - h[t]**2) * dh #???

            #find updates for h
            #输入层->中间层，中间层->中间层的权重变化
            #以及中间层偏置量的变化
            dW_xh += np.dot(dh_raw, xhat[t].T)
            dW_hh += np.dot(dh_raw, h[t-1].T)
            db_h += dh_raw

            #save dh_next for subsequent iteration
            dh_next = np.dot(self.W_hh.T, dh_raw)

        for dparam in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(dparam, -5, 5, out=dparam)#clip to mitigate exploding gradients
        # clip函数将数组内的值限定在[-5,5]之间，<-5的元素值被设定为-5，>5的元素值被设定为5
        # out=dparam输出给dparam
        #update RNN parameters according to Adagrad
        # zip()函数将数组按规律变成一个tuple(),便于更新操作
        for param, dparam, adaparam in zip([self.W_hh, self.W_xh, self.W_hy, self.b_h, self.b_y],
                                [dW_hh, dW_xh, dW_hy, db_h, db_y],
                                [self.adaW_hh, self.adaW_xh, self.adaW_hy, self.adab_h, self.adab_y]):
            adaparam += dparam*dparam
            param += -self.learning_rate*dparam/np.sqrt(adaparam+1e-8)
        #为下一个batch-size的训练数据准备h
        self.h = h[len(x)-1]

        return loss

    #let the RNN generate text
    def sample(self, seed, n):
        ndxs = []
        h = self.h

        xhat = np.zeros((self.insize, 1))
        xhat[seed] = 1#transform to 1-of-k

        for t in range(n):
            h = np.tanh(np.dot(self.W_xh, xhat) + np.dot(self.W_hh, h) + self.b_h)#update the state
            y = np.dot(self.W_hy, h) + self.b_y
            p = np.exp(y) / np.sum(np.exp(y))
            # select a int number according to their probabilities stored in variable p.
            ndx = np.random.choice(range(self.insize), p=p.ravel())


            xhat = np.zeros((self.insize, 1))
            xhat[ndx] = 1

            ndxs.append(ndx)

        return ndxs


def test():# test a very very long txt file.
    #open a text file
    data = open("..//DataSet//rnn_test.txt", 'r').read() # should be simple plain text file
    # set()查找data里所有不重复的字符
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print ("data has %d characters, %d unique." %(data_size, vocab_size))

    #make some dictionaries for encoding and decoding from 1-of-k
    #双向映射的字典，例如：char_to_ix["c"]=0, 则ix_to_char[0]="c"

    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }

    #insize and outsize are len(chars). hidsize is 100. seq_length is 25. learning_rate is 0.1.
    rnn = RNN(len(chars), len(chars), 100, 0.1)

    #iterate over batches of input and target output
    seq_length = 20
    losses = []
    #简便的计算seq_length次smooth_loss
    smooth_loss = -np.log(1.0/len(chars))*seq_length#loss at iteration 0
    losses.append(smooth_loss)



    for i in range(int(len(data)/seq_length)):
        x = [char_to_ix[c] for c in data[i*seq_length:(i+1)*seq_length]]#inputs to the RNN
        y = [char_to_ix[c] for c in data[i*seq_length+1:(i+1)*seq_length+1]]#the targets it should be outputting

        if i%1000==0:
            sample_ix = rnn.sample(x[0], 200)
            txt = ''.join([ix_to_char[n] for n in sample_ix])
            print("\n")
            print(txt)

        loss = rnn.train(x, y)

        #smooth_loss = smooth_loss*0.999 + loss*0.001
        smooth_loss=loss

        if i%1000==0:
            print ("iteration %d, smooth_loss = %f" % (i, smooth_loss))
            losses.append(smooth_loss)


    plt.plot(range(len(losses)), losses, 'b', label='smooth loss')
    plt.xlabel('time in thousands of iterations')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def test1(): # test a short txt file.
    # open a text file
    data = open('DataSet//wangfeng.txt', 'r').read()  # should be simple plain text file
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print("data has %d characters, %d unique." % (data_size, vocab_size))

    # make some dictionaries for encoding and decoding from 1-of-k
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    # insize and outsize are len(chars). hidsize is 100. seq_length is 25. learning_rate is 0.1.
    rnn = RNN(len(chars), len(chars), 100, 0.1)

    # iterate over batches of input and target output
    seq_length = 25
    losses = []
    smooth_loss = -np.log(1.0 / len(chars)) * seq_length  # loss at iteration 0
    losses.append(smooth_loss)

    i = 0
    count = 0
    while True:
        # for i in range(int(len(data)/seq_length)):
        x = [char_to_ix[c] for c in data[i * seq_length:(i + 1) * seq_length]]  # inputs to the RNN
        y = [char_to_ix[c] for c in
             data[i * seq_length + 1:(i + 1) * seq_length + 1]]  # the targets it should be outputting
        if i % (int(len(data) / seq_length) - 1) == 0:
            i = 0

        if i % 1000 == 0:
            sample_ix = rnn.sample(x[0], 200)
            txt = ''.join([ix_to_char[n] for n in sample_ix])
            print("\n")
            print(txt)

        loss = rnn.train(x, y)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        if i % 1000 == 0:
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
    test()