
# 作者：yeqiang
# email:qqiangye@gmail.com
# 这是一个定义中文汉字字典的文件，包括节点和路径两个类，
# 分别对应于图的节点和边。
# 用于搜集汉语各单字在组成词语或句子时的结合特点
# 便于后续进一步分析
import numpy as np
import warnings


# 定义字典内的基本节点类
class BaseNode(object):
  #cur_id = 1
  def __init__(self, text):
    #self.nid = self.__class__.cur_id    # 节点编号,例如"001"
    #self.__class__.cur_id += 1
    self.text = text  # 节点内容,例如"叶"
    self.count = 0       # 节点一共出现的次数
    self.post_paths = {}  # 节点后续链接路径形成的字典结构，键值是text->text
    self.pre_paths = {}   # 节点前接连接路径形成的字典结构，键值是text->text
    self.pre_path_count = 0 # 该节点作为后节点的所有路径数，不等于该节点拥有的所有前节点数
                            # 而是该节点的所有前节点下的路径数之和，下同
    self.post_path_count = 0  # 这两个变量在路径被遍历一次时增加1



  def touch(self):
    '''
    一个节点被touch一次
    '''
    self.count += 1

  # 查找连接到post_n的路径是否存在，存在则将其返回，不存在则返回None
  def search_path_to(self,post_n):
    key = create_path_key(self, post_n)
    path = self.post_paths.get(key)  # if not exist in dict, will return None
    return path   # if not found return None, if found return target.

  # 查找从pre_n链接来的path是否存在，存在则将其返回，不存在则返回None
  def search_path_from(self,pre_n):
    key = create_path_key(pre_n, self)
    obj_p = self.pre_paths.get(key)
    return obj_p  # if not found return None, if found return target.

  def new_path_to(self,post_n):
    new_p = BasePath(self, post_n)
    self.post_paths[new_p.pid] = new_p  # pid as a key value of post_paths dictionary.
    post_n.pre_paths[new_p.pid] = new_p
    return new_p

  # 连接至另一个节点，如果已有链接则不新建链接。返回链接
  def connect_to(self,post_n):
    if post_n is None:
      return None
    obj_p = self.search_path_to(post_n)
    if obj_p is not None:
      return obj_p
    return self.new_path_to(post_n)

  def new_path_from(self,pre_n):
    new_p = BasePath(pre_n, self)
    self.pre_paths[new_p.pid] = new_p
    pre_n.post_paths[new_p.pid] = new_p
    return new_p

  def connect_from(self,pre_n):
    if pre_n is None:
      return None
    obj_p = self.search_path_from(pre_n)
    if obj_p is not None:
      return obj_p
    return self.new_path_from(pre_n)

  # 删除一个路径
  def disconnect_to(self,post_n):
    obj_p = self.search_path_to(post_n)
    if obj_p is not None:
      self.post_paths.pop(obj_p.pid)
      post_n.pre_paths.pop(obj_p.pid)
    return obj_p
  # 断开一个路径
  def disconnect_from(self,pre_n):
    obj_p = self.search_path_from(pre_n)
    if obj_p is not None:
      self.pre_paths.pop(obj_p.pid)
      pre_n.post_paths.pop(obj_p.pid)
    return obj_p

  def __str__(self):
    s = str(self.text) + " (" + str(len(self.pre_paths))+"," + str(len(self.post_paths)) + ")"
    return s


  def show_brief(self):
    print("- Information of BaseNode '%s': %d(%d) ->'%s'-> %d(%d) -"
          % (self.text, len(self.pre_paths),self.pre_path_count,
             self.text, len(self.post_paths),self.post_path_count))

  def show_paths(self,paths):
    result = "\t"
    i = 0
    for key in paths:
      result += str(paths[key]) + "\t"
      i += 1
      if i % 5 == 0:
        i = 0
        print(result)
        result = "\t"
    print(result)


  # 打印节点内容
  def show_detail(self):
    self.show_brief()
    print("\tpre_paths:")
    self.show_paths(self.pre_paths)
    print("\tpost_paths:")
    self.show_paths(self.post_paths)
    print("- End of BaseNode Information -")

# 定义一个生成路径key的函数，路径key既是节点类里路径字典内的key
# 同时也是对应路径的id
def create_path_key(pre_n, post_n):
  """
  根据一个路径的前后节点生成一个唯一确定路径的Key

  :param pre_n:
  :param post_n:
  :return:
  """
  return pre_n.text + "->" + post_n.text

# 定义字典内路径
class BasePath(object):
  def __init__(self, pre_n=None, post_n=None):
    self.pid = create_path_key(pre_n, post_n) # path id
    self.pre_n = pre_n        # 路径的左端节点
    self.post_n = post_n      # 路径的右段节点
    # self.records = {}         # 路径被使用过的记录，记录的事词语或断句的编号及其通过次数
    self.count = 0       # 路径被访问次数

  # 路径会记录最近一定次数的经过该路径的词语、短语或句子
  # sid 是对某一个词语或句子的hash编码，作为记录字典的键值
  def touch(self):
    """
    路径访问次数+1
    :return:
    """
    self.count += 1
    self.pre_n.post_path_count += 1
    self.post_n.pre_path_count += 1

  def __str__(self):
    s = "{}({})".format(str(self.pid),self.count)
    return s

  def show_detail(self):
    print("- Detail of the path: {} -".format(self.pid))
    print(self)


def test():
  char_dict = {}
  phrase="我错了你也做错事情了我们是错上加错啊"
  c=phrase[0]
  pre_node = char_dict.get(c)
  if pre_node is None:
    pre_node = BaseNode(c)
    char_dict[c] = pre_node
    pre_node.touch()
  l = len(phrase)
  for i in range(1,l-1):
    c = phrase[i]
    post_node = char_dict.get(c)
    if post_node is None:
      post_node = BaseNode(c)
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
