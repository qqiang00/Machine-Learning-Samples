from crf_graphmodel import *
import numpy as np
import math
import char_dfa as cdfa
try:
  from xml.etree.cElementTree import Element, SubElement, ElementTree
except ImportError:
  from xml.etree.ElementTree import Element, SubElement, ElementTree
import datetime

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

def list2string(f_list,sep=" "):
  """
  给出一个float一维数据数组，返回拥有该数组的字符串形式，用
  :param f_list:
  :param sep: 分隔符
  :return:
  """
  result = ""
  for i in range(len(f_list)):
    result+=str(f_list[i])
    if i< len(f_list)-1:
      result += sep
  return result

def string2list(source,sep=" "):
  """

  :param source:
  :param sep:
  :return:
  """
  l=source.split(sep)
  for i in range(len(l)):
    l[i] = float(l[i])
  return l

def matrix2string(matrix,sep=" "):
  matrix.shape = -1,1
  size = matrix.size
  result = ""
  for i in range(size):
    result += str(matrix[i][0])
    if i < size-1:
      result += sep
  return result

def string2matrix(source, sep=" ",shape=(4,4)):
  l = string2list(source)
  matrix = np.asarray(l,dtype=float)
  matrix.shape=shape
  return matrix



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



class CharDict(object):
  char_dict = {}
  author =""
  create_date =""
  description =""
  @classmethod
  def reset_dict(cls):
    cls.char_dict={}
    cls.author = ""
    cls.create_date = ""
    cls.description = ""

  @classmethod
  def __init__(cls):
    cls.reset_dict()

  @classmethod
  def show_dict_info(cls):
    print("Author:{}".format(cls.author))
    print("Create Date:{}".format(cls.create_date))
    print("Description:{}".format(cls.description))
    print("Total Nodes:{}".format(len(cls.char_dict.keys())))

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

  @staticmethod
  def _analysis_results(obs,results) -> tuple:
    tuple_result = []
    final = len(results) -1
    final_result = results[final]

    max_item = max(final_result)
    pre_state = max_item[1]

    max_item_state = final_result.index(max_item)

    tuple_result.insert(0,(obs[final],char_status2[str(max_item_state)]))
    tuple_result.insert(0,(obs[final-1],char_status2[str(pre_state)]))

    for t in range(final - 1, 1, -1):
      _, pre_state = results[t][pre_state]
      tuple_result.insert(0, (obs[t-1],char_status2[str(pre_state)]))
    return tuple_result

  @classmethod
  def segment(cls,sentence):
    """
    给一个完全由中文组成的句子进行分词
    :param sentence: 一个经过预处理没有标点符号全部由中文汉字组成的短语或句子
    :return: 一个list,其内每一个元素是一个tuple(a,b)，a是汉子，b是其状态，取BMES其一
    """
    sentence = _hat_string(sentence) # 给待分析的句子加上头尾标志.例如："我喜欢你"变成"^我喜欢你$"
    length = len(sentence)  # length>=2
    if length == 2:
      return ""
    results = [[(0,None),(0,None),(0,None),(1,None)]] # "^"的状态

    pre_n = cls.get_char_node(sentence[0])  # 节点："^"
    cur_n = cls.get_char_node(sentence[1])  # 节点："我"

    pre_p = pre_n.search_path_to(cur_n) # 路径 从 "^"->"我"

    for t in range(1,length-1):   # 从"我"开始分析。结尾标志"$"不分析
      results.append([])   # 针对一个字，增加一个list，形式是[(p,s),(p,s),(p,s),(p,s)]
      post_n = cls.get_char_node(sentence[t+1]) # 获取节点t+1："喜"

      for s in range(4):  # "BMES"中的每一个状态
        prob, pre_s = max([(results[t-1][s0][0]*pre_n.factor_s_to_s(s0,s),s0) for s0 in range(4)])
        value = prob

        value += cur_n.factor_s(s)
        if pre_p is not None:
          value += pre_p.factor_o_to_s(s)
        post_p = cur_n.search_path_to(post_n)
        if post_p is not None:
          value += post_p.factor_s_to_o(s)

        results[t].append((value,pre_s))
      pre_n,cur_n = cur_n,post_n
      pre_p = post_p


    return CharDict._analysis_results(sentence,results)

  @classmethod
  def displayDicInfo(cls):
    print("dictionary brief info:")
    print("\tauthor:%s"%cls.author)
    print("\tcreate date:%s"%cls.create_date)
    print("\tdescription:%s"%cls.description)
    print("\t%d nodes in the dictionary"%(len(cls.char_dict)))

  @staticmethod
  def node2ele(cur_n,nodeEle):
    nodeEle.set("key", cur_n.text)
    nodeEle.set("count", str(cur_n.count))
    nodeEle.set("s_list", list2string(cur_n.s_list))
    nodeEle.set("s_matrix", matrix2string(cur_n.s_matrix))

  @staticmethod
  def ele2node(nodeEle,cur_n):
    cur_n.text = nodeEle.get("key")
    cur_n.count = int(nodeEle.get("count"))
    cur_n.s_list = string2list(nodeEle.get("s_list"))
    cur_n.s_matrix = string2matrix(nodeEle.get("s_matrix"))

  @staticmethod
  def path2ele(cur_p,pathEle):
    pathEle.set("count", str(cur_p.count))
    pathEle.set("pre_node", cur_p.pre_n.text)
    pathEle.set("post_node", cur_p.post_n.text)
    pathEle.set("s_matrix", matrix2string(cur_p.s_matrix))

  @staticmethod
  def ele2path(pathEle,cur_p):
    cur_p.count = int(pathEle.attrib["count"])
    cur_p.s_matrix = string2matrix(pathEle.attrib["s_matrix"])

  @classmethod
  # save dict to an xml file
  def saveDictToXml(cls, file_name):
    dictEle = Element("dict")
    dictEle.set("description", cls.description)
    dictEle.set("count", str(len(cls.char_dict)))
    dictEle.set("author", cls.author)
    cur_time = str(datetime.datetime.now())
    dictEle.set("create_date", cur_time)

    for key in cls.char_dict:
      cur_n = cls.char_dict[key]    # 一个节点
      nodeEle = SubElement(dictEle, "node") # 建立一个xmlEle
      CharDict.node2ele(cur_n,nodeEle)  # 关联xmlEle

      prePathsEle = SubElement(nodeEle, "pre_paths")
      prePathsEle.set("count", str(len(cur_n.pre_paths)))
      for path_key in cur_n.pre_paths:
        prePathEle = SubElement(prePathsEle, "path")
        pre_path = cur_n.pre_paths[path_key]
        CharDict.path2ele(pre_path,prePathEle)

      postPathsEle = SubElement(nodeEle, "post_paths")
      postPathsEle.set("count", str(len(cur_n.post_paths)))
      for path_key in cur_n.post_paths:
        postPathEle = SubElement(postPathsEle, "path")
        post_path = cur_n.post_paths[path_key]
        CharDict.path2ele(post_path,postPathEle)

    tree = ElementTree(dictEle)

    try:
      tree.write(file_name, encoding="utf-8", xml_declaration="version = 1.0")
      print("dictionary successfully saved in: %s"%(file_name))
    except Exception:
      print("error occurs when write to xml file")

  @classmethod
  def loadDictFromXml(cls,file_name):

    cls.char_dict = {}  # reset dict, add root and end nodes
    tree = ElementTree(file=file_name)
    dictEle = tree.getroot()
    cls.author = dictEle.attrib["author"]
    cls.description = dictEle.attrib["description"]
    cls.create_date = dictEle.attrib["create_date"]
    node_count_from_attrib = int(dictEle.attrib["count"])
    for nodeEle in dictEle:  # child node
      n_key = nodeEle.get("key")
      node = CharDict.get_char_node(n_key)
      CharDict.ele2node(nodeEle,node)

      for pathsEle in nodeEle:  # pre_paths and post_paths
        for pathEle in pathsEle:
          pre_node = CharDict.get_char_node(pathEle.get("pre_node"))
          post_node = CharDict.get_char_node(pathEle.get("post_node"))
          path = pre_node.connect_to(post_node)
          CharDict.ele2path(pathEle,path)

    node_count_cal = len(cls.char_dict)
    if node_count_from_attrib == node_count_cal:
      print("dictionary successfully loaded.")
    else:
      print("count in xml file(%d) dismatch real node count(%d)" %
            node_count_from_attrib, node_count_cal)
    # self.displayDicInfo()

    #except Exception as e:
    #  print("error in loading dict from: %s"%file_name)
    #  print(e)
    #finally:
    #  pass

if __name__ == "__main__":
  #learning_file = "/home/yeqiang/chinese_resources/pku_training.utf8"
  dict_file = "/home/yeqiang/chinese_resources/pku_training_dict.xml"

  #result=mark_char_status("不错 的 东西 我 错 了 出错 错误 错上加错 "," ")
  #CharDict.build_dict_from_file(learning_file,sep=" ")
  train_str = \
    "我们 是 共产主义 接班 人 中华 人民 共和国 " \
    "这 件 事情 你 做 得 不错 我们 都 没有 犯 错误 " \
    "因为 之前 已经 犯 过 类似 错误 了 " \
    "要 亡羊补牢 而 不是 错上加错 " \
    "这 是 你 的 东西 吗 " \
    "我 的 在 哪里 " \
    "你 出去 帮 我 把 门 关 起来 " \
    "奥巴马 总统 特朗普 先生 奥巴马 奥巴马 错上加错 错上加错 错上加错 " \
    "喜欢 这样 喜欢 亡羊补牢 共产党 我们 国家 " \
    "喜欢 这样 喜欢 亡羊补牢 共产党 我们 国家 共产党 领导 共产党 领导 中国 中国 中国"
  train_tuple = mark_char_status(train_str," ")
  #CharDict.build_dict(train_tuple)
  #CharDict.show_dict_info()
  #CharDict.author="YeQiang"
  #CharDict.description="用于分词的字典"
  #CharDict.saveDictToXml(dict_file)
  #CharDict.reset_dict()
  #CharDict.show_dict_info()
  print("loading dic from xml file...")
  CharDict.loadDictFromXml(dict_file)
  print("loading completed!")
  CharDict.show_dict_info()

  target_string="希腊的经济结构比较单一"
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