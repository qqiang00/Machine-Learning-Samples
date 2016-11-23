from crf_graphmodel import *
import numpy as np
import math
import char_dfa as cdfa

def mark_char_status(source:str,seperator=" "):
  '''
  将一个已经做过分词的句子（或短语）中每一个字在词语中的位置状态标记出来
  :param source: 一个已经分词好了的句子或短语
  :param seperator: 分词用的分隔符，默认为空格
  :return: 一个tuple的列表[(c,s),(c,s),...],c为字，s为其状态，取值为"B","M","E"或"S"
  '''
  wordlist = source.split(seperator)
  result = []
  for word in wordlist:
    l = len(word)
    if l == 1:
      result.append((word[0],"S")) # 单字词
    elif l == 2:
      result.append((word[0],"B"))  # 双字词首字
      result.append((word[1],"E"))  # 双字词末字
    else:
      for i,c in enumerate(word):
        if i == 0:
          result.append((c,"B"))
        elif 0 < i < l-1:
          result.append((c,"M"))
        else:
          result.append((c,"E"))
  return result

def de_mark_char_status(source:list):
  """
  将一个标记了的短语按照标记生成词列表
  :param source:
  :return: 生成好的词列表
  """
  result = []
  word = ""
  for c,s in source:
    if s == "B":
      if word is not "":
        result.append(word)
      word = c
    elif s == "S":
      if word is not "":
        result.append(word)
        word=""
      result.append(c)
    elif s == "M":
      word += c
    else: # s== "E"
      word += c
      result.append(word)
      word=""
  if word is not "":
    result.append(word)
  return result



def _hat_tuple_list(source):
  """
  给一个已经有状态标记的数据list首尾添加帽子。

  便于统计某一个单字位于词首、或词尾的次数
  :param tuple_list_source:
  :return:
  """
  start = ("^","S")
  end = ("$","S")
  source.insert(0,start)
  source.append(end)
  return source

def _hat_string(source):
  """
  给一个待分析的中文字符串添加首尾帽子
  :param source:
  :return:
  """
  return "^"+source+"$"

def frequency(matrix,pre=-1,post=-1):
  """
  返回矩阵中特定元素的和

  :param matrix:  要分析的矩阵
  :param i: 行索引，对应状态转化矩阵是前一个汉子的信息
  :param j: 列索引，对应状态转化矩阵是后一个汉字的信息
  :param lambda_: 平滑因子，牺牲大频数的概率，适当增加小频数的概率
  :return:
  """
  if pre == -1 and post == -1:
    return np.sum(matrix)

  i,j = matrix.shape
  if pre <= -1:
    sub_sum = np.sum(matrix[:,post])
  elif post <= -1:
    sub_sum = np.sum(matrix[pre,:])
  elif pre < i and post < j:
    sub_sum = matrix[pre,post]
  else:
    sub_sum = 0.0
  #return math.pow(sub_sum,lambda_)/math.pow(m_sum,lambda_)
  return sub_sum

class CharDict(object):
  char_dict = {}

  def __init__(self):
    self.author=""
    self.create_date=""
    self.description=""

  @classmethod
  def show_dict_info(cls):
    print("Nodes:{}".format(len(cls.char_dict.keys())))
    for key in cls.char_dict.keys():
      node = cls.char_dict[key].show_brief()

  @classmethod
  def load_root_node(cls):
    cls.char_dict = {"^": CRFNode("^"), "$": CRFNode("$")}

  @classmethod
  def build_dict(cls, tuple_source):
    '''
    用一个以seperator作为分隔符的以分好词的字符串扩充（更新）字典信息
    :param tuple_source:List表示的短语列表，每个元素是一个tuple:(c,s),c表示字本身，s表示其位置状态信息
    :return:更新字典信息
    '''
    source = _hat_tuple_list(tuple_source) # 让source 包含头尾信息
    pre_c, pre_s = source[0]  # source[0] 一定是("^","S")
    pre_n = cls.get_char_node(pre_c)
    pre_n.touch_as(pre_s)
    length=len(source)

    for i in range(1,length):
      post_c, post_s = source[i]
      #print("Learning:{} | {}".format(post_c,post_s))

      post_n = cls.get_char_node(post_c)

      post_n.touch_as(post_s)
      pre_n.modify_matrix(pre_s,post_s)

      con = pre_n.connect_to(post_n)
      con.touch_as(pre_s,post_s)

      pre_n, pre_s = post_n, post_s
      pass

  @classmethod
  def build_dict_from_file(cls,file,sep=" "):
    """
    从一个做好分隔的txt文件中
    :param file:
    :param sep:
    :return:
    """
    # learn chinese char from a file.
    # 从一个文本文件学习构建字典，文件可以是中英文混杂，要求是utf-8格式文件
    # 如果不是utf-8格式将忽视
    # txttencodeconverter 可以批量将常见其他编码格式的文本转化为utf-8文本
    # times 为重复学习的次数
    file_name = file
    print("learning:%s" % file_name)
    f = open(file_name, "r", encoding="utf-8")
    is_end = False
    try:
      while not is_end:
        content = f.readline(200)
        #print(content)
        if(len(content)) == 0:
          break
        # pp 内包含一个有限自动机，用来预处理文本：分开中文、英文、标点、数字到不同的数组里
        #str_list = cdfa.pre_segment(content)
        #for s_type, text in str_list:  # 数组元素是一个二元tuple:(a,b).
        #  if s_type == "cn":
        text=content
        tuple_source = mark_char_status(text)
        #print("Learning:"+str(tuple_source))
        cls.build_dict(tuple_source)

    except UnicodeDecodeError:
      print("Can't decode file:%s" % file_name)
    except IOError:
      print("IOError:\n")
    finally:
      f.close()

  @classmethod
  def get_char_node(cls,char="^"):
    """
    确保能获取一个节点，如果没有则在字典里新建一个
    :param char:
    :return:
    """
    if char is None or len(char) != 1:
      char = "^"
    node = cls.char_dict.get(char)
    if node is None:
      node = CRFNode(char)
      cls.char_dict[char] = node  # add new node to dict
    return node

  @classmethod
  def segment(cls,sentence):
    """
    给一个完全由中文组成的句子进行分词
    :param sentence: 一个经过预处理没有标点符号全部由中文汉字组成的短语或句子
    :return: 一个list,其内每一个元素是一个tuple(a,b)，a是汉子，b是其状态，取BMES其一
    """
    sentence = _hat_string(sentence) # 给待分析的句子加上头尾标志

    length = len(sentence)  #
    out_matrix = np.zeros((length,4)) # 建立一个矩阵
    tuple_list = []
    pre_n = cls.get_char_node(sentence[0])
    cur_n = cls.get_char_node(sentence[1])

    pre_p = pre_n.search_path_to(cur_n)
    out_matrix[0][3] = 1
    out_matrix[length-1][3] = 1
    index_pre_max = 3
    for i in range(1,length-1):   # 首尾不需要分析标记，都为"S"

      post_n = cls.get_char_node(sentence[i+1])
      # TODO
      for j in range(4):   # 针对当前节点每一个状态"BMES"
        value = 0
        #temp = [0,0,0,0]
        #k_sum = np.sum(pre_n.s_matrix[:, j])
        #k_sum = k_sum if k_sum > 0 else 1
        #for k in range(4):
          # 前一个字符在状态k的概率， 前一个节点的k状态转至当前状态j的概率
        #  temp[k] = out_matrix[i-1,k]*(pre_n.s_matrix[k,j]/k_sum)
        #value += max(temp)
        #print("temp")
        #print(temp)
        k = index_pre_max
        k_sum = np.sum(pre_n.s_matrix[k,:])
        k_sum = k_sum if k_sum>0 else 1
        value += out_matrix[i-1][k]*pre_n.s_matrix[k,j]/k_sum

        value + cur_n.s_list[j]/cur_n.count

        if pre_p is not None:
          value += frequency(pre_p.s_matrix,post=j)/cur_n.pre_path_count
        post_p = cur_n.search_path_to(post_n)
        if post_p is not None:
          value += frequency(post_p.s_matrix,pre=j)/cur_n.post_path_count
        out_matrix[i,j] = value

      index_pre_max = out_matrix[i].argmax(0)
      tuple_list.append((sentence[i],char_status2[str(index_pre_max)]))

      pre_n,cur_n = cur_n,post_n
      pre_p = post_p
    print("transit matrix:")
    print(out_matrix)
    return tuple_list


if __name__ == "__main__":
  learning_file = "/home/yeqiang/chinese_resources/pku_training.utf8"
  result=mark_char_status("不错 的 东西 我 错 了 出错 错误 错上加错 "," ")
  CharDict.build_dict_from_file(learning_file,sep=" ")
  train_str = \
    "我们 是 共产主义 接班 人 中华 人民 共和国 " \
    "这 件 事情 你 做 得 不错 我们 都 没有 犯 错误 " \
    "因为 之前 已经 犯 过 类似 错误 了 " \
    "要 亡羊补牢 而 不是 错上加错 " \
    "这 是 你 的 东西 吗 " \
    "我 的 在 哪里 " \
    "你 出去 帮 我 把 门 关 起来 "
  #train_tuple = mark_char_status(train_str," ")
  #CharDict.build_dict(train_tuple)
  #CharDict.show_dict_info()
  target_string="不错的东西我错了出错了错上加错你的东西在哪里"
  result = CharDict.segment(target_string)
  print(result)
  result = de_mark_char_status(result)
  print(result)
  while True:
    target_string=input("type your sentence:")
    if len(target_string) <=0:
      break
    result = CharDict.segment(target_string)
    print(result)
    result = de_mark_char_status(result)
    print(result)

  while True:
    c=input("type 2 nodes to search path:")
    if len(c) <2:
      break
    node = CharDict.char_dict[c[0]]
    node2=CharDict.char_dict[c[1]]
    node.show_detail()
    node2.show_detail()
    path2 = node.search_path_to(node2)
    if path2 is not None:
      path2.show_detail()
    else:
      print("Path not exist")