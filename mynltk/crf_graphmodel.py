
# 作者：yeqiang
# email:qqiangye@gmail.com
# 这是一个适用于条件随机场的节点和路径类，包括节点和路径两个类，
# 分别对应于图的节点和边。
# 用于搜集汉语各单字在组成词语或句子时的结合特点
# 便于后续进一步分析
import numpy as np
from graphmodel import BaseNode, BasePath, create_path_key
import strtool

try:
  from xml.etree.cElementTree import Element, SubElement, ElementTree
except ImportError:
  from xml.etree.ElementTree import Element, SubElement, ElementTree

state2index = {"B":0,"M":1,"E":2,"S":3} #记录个状态在list内的索引
index2state = {"0":"B","1":"M","2":"E","3":"S"} #记录索引对应的状态字符串



def node2ele(cur_n, nodeEle):
  nodeEle.set("key", cur_n.text)
  nodeEle.set("count", str(cur_n.count))
  nodeEle.set("s_list", strtool.list2string(cur_n.s_list))
  nodeEle.set("s_matrix", strtool.matrix2string(cur_n.s_matrix))



def ele2node(nodeEle, cur_n):
  cur_n.text = nodeEle.get("key")
  cur_n.count = int(nodeEle.get("count"))
  cur_n.s_list = strtool.string2list(nodeEle.get("s_list"))
  cur_n.s_matrix = strtool.string2matrix(nodeEle.get("s_matrix"))

def path2ele(cur_p, pathEle,is_post_path = True):
  pathEle.set("count", str(cur_p.count))
  if is_post_path is True:
    pathEle.set("post_node", cur_p.post_n.text)
  else:
    pathEle.set("pre_node", cur_p.pre_n.text)
  pathEle.set("s_matrix", strtool.matrix2string(cur_p.s_matrix))

def ele2path(pathEle, cur_p):
  cur_p.count = int(pathEle.attrib["count"])
  cur_p.s_matrix = strtool.string2matrix(pathEle.attrib["s_matrix"])

def show_status_matrix(matrix):
  print("\tTransfer Matrix:")
  print("\tPre\Post\t\tB\t\t\tM\t\t\tE\t\t\tS")
  for i in range(4):
    s = "\t   " + index2state[str(i)]
    for j in range(4):
      s += "\t{:>10s}".format(str(matrix[i, j]))
    print(s)

# 定义字典内的节点类
class CRFNode(BaseNode):
  #cur_id = 1
  def __init__(self, text):
    super().__init__(text)
    self.s_list = [0,0,0,0] # 节点所代表的字依次作为词首、词中、词尾及单字词出现的次数
    self.s_matrix = np.zeros((4,4))  # 状态转化矩阵

  def touch_as(self,status):
    '''
    一个节点以status这个状态被touch一次
    :param status: 是字在词语里的位置信息
    :return:
    '''
    index = state2index[status]
    # if index is None:
    #   warnings.warn("wrong status for a character in a phrase",RuntimeWarning)
    self.s_list[index] += 1
    #self.count += 1  与下句效果相同
    super().touch()

  def modify_matrix(self,pre_s,post_s):
    self.s_matrix[state2index[pre_s],state2index[post_s]] += 1

  def factor_s(self,status:int):
    c_sum = self.count if self.count > 0 else 0.0001
    return self.s_list[status]/c_sum

  def factor_s_to_s(self,s0:int,s1:int):
    c_sum = self.count if self.count > 0 else 0.0001
    return self.s_matrix[s0,s1]/c_sum

  #def __str__(self):
  #  s = str(self.content) + " (" + str(len(self.prepaths))+"," + str(len(self.postpaths)) + ")"
  #  return s
  def new_path_from(self,pre_n):
    new_p = CRFPath(pre_n, self)
    self.pre_paths[new_p.pid] = new_p
    pre_n.post_paths[new_p.pid] = new_p
    return new_p

  def new_path_to(self,post_n):
    new_p = CRFPath(self, post_n)
    self.post_paths[new_p.pid] = new_p  # pid as a key value of post_paths dictionary.
    post_n.pre_paths[new_p.pid] = new_p
    return new_p

  def show_status_list(self):
    # for key in self.prepaths:
    #  pre_nchars+=self.prepaths[key].pre_n.content
    #  pre_nchars+=" "
    # print("  Pre Nodes(%d): %s" % (len(self.prepaths), pre_nchars))
    print("\tCounts:%d ['B':%d,'M':%d,'E':%d,'S':%d]"
          % (self.count, self.s_list[0], self.s_list[1],
             self.s_list[2], self.s_list[3]))

  # 打印节点内容
  def show_detail(self):
    #pre_nchars=""
    super().show_brief()
    self.show_status_list()
    show_status_matrix(self.s_matrix)
    print("\tPre paths:")
    self.show_paths(self.pre_paths)
    print("\tPost paths:")
    self.show_paths(self.post_paths)
    print("- End of Node Information -")


# 定义字典内路径
class CRFPath(BasePath):
  def __init__(self, pre_n=None, post_n=None):
    super().__init__(pre_n,post_n)
    self.s_matrix = np.zeros((4,4))  # 记录链接两个字的路径分别作为不同状态是的次数

  def touch_as(self, pre_s, post_s):
    super().touch()
    self.s_matrix[state2index[pre_s],state2index[post_s]] += 1
    #self.touch()  # self.touch与super().touch()的效果一样，将节点访问次数+1


  #def __str__(self):
  #  return super().__str__()
  def factor_s_to_o(self,state):
    c_sum = self.count if self.count > 0 else 0.0001
    return np.sum(self.s_matrix[state,:])/c_sum

  def factor_o_to_s(self,state):
    c_sum = self.count if self.count > 0 else 0.0001
    return np.sum(self.s_matrix[:,state]) / c_sum

  def show_detail(self):
    super().show_detail()
    show_status_matrix(self.s_matrix)
    #
    #for key in self.records:
    #  s = "      \"" + str(key) + "\": (" + str(self.records[key]) + ")"
    #  print(s)


def test():
  char_dict = {}
  phrase="我错了你也做错事情了我们是错上加错啊"
  c=phrase[0]
  pre_node = char_dict.get(c)
  if pre_node is None:
    pre_node = CRFNode(c)
    char_dict[c] = pre_node
    pre_node.touch()
  l = len(phrase)
  for i in range(1,l-1):
    c = phrase[i]
    post_node = char_dict.get(c)
    if post_node is None:
      post_node = CRFNode(c)
      char_dict[c] = post_node
      post_node.touch()

    o_path = pre_node.connect_to(post_node)
    o_path.touch()

    pre_node = post_node


  node = char_dict["错"]
  pre_node = char_dict["了"]
  post_node = char_dict["你"]
  print("string is:'{}'".format(phrase))
  node.show_detail()
  pre_node.show_detail()
  post_node.show_detail()

  path=pre_node.search_path_to(post_node)
  print("there is a path from {} to {}:".format(pre_node.text,post_node.text))
  path.show_detail()
  print("now disconnect it")
  pre_node.disconnect_to(post_node)

  path=pre_node.search_path_to(post_node)
  if path is None:
    print("now no path between them")
  else:
    path.show_detail()
  print("information of the two nodes")
  pre_node.show_detail()
  post_node.show_detail()
  print("now re connect it using connectfrom")
  post_node.connect_from(pre_node)
  print("now the path infomation:")
  path1=pre_node.search_path_to(post_node)
  path1.show_detail()
  path2=post_node.search_path_from(pre_node)
  path2.show_detail()
  if path1 is path2:
    print("two path are the same")
  pre_node.show_detail()
  post_node.show_detail()



if __name__ == "__main__":
  test()