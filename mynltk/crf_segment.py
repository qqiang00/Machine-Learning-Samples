import crf_graphmodel as crfgm
import filetool
import os

try:
  from xml.etree.cElementTree import Element, SubElement, ElementTree
except ImportError:
  from xml.etree.ElementTree import Element, SubElement, ElementTree
import datetime

def mark_char_status(source:str, sep=' '):
  '''
  对一个以特定分隔符分割的已做好分词的短语语料中的每一个字进行BMES分词标记
  :param source: 一个已经分词好了的句子或短语
  :param sep: 分词用的分隔符，默认为空格
  :return: 一个tuple的列表[(c,s),(c,s),...],c为字，s为其状态，取值为"B","M","E"或"S"
  '''
  w_list = source.split(sep)
  result = []
  for word in w_list:
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

def de_mark_char_status(source:list,sep=' '):
  """
  将一个标记了的短语按照标记生成词列表
  :param source:
  :param sep:
  :return:
  """
  result = ""
  word = ""
  for c,s in source:
    result += c
    if s == "S" or s == "E":
      result += sep
  l=len(result)
  if result.endswith(sep):
    result = result[0:l-len(sep)]
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

def _analysis_results(obs, results) -> list:
  """
  将粉刺结果以list[tuple]的形式表现出来
  :param obs: 观测（带分词的短语）
  :param results: 已经分好词的数据
  :return: list,每个元素是tuple(c,s),c是文本，s是BMES状态
  """
  tuple_result = []
  final = len(results) - 1
  final_result = results[final]
  max_item = max(final_result)
  pre_state = max_item[1]
  max_item_state = final_result.index(max_item)

  tuple_result.insert(0, (obs[final], crfgm.index2state[str(max_item_state)]))
  tuple_result.insert(0, (obs[final - 1], crfgm.index2state[str(pre_state)]))

  for t in range(final - 1, 1, -1):
    _, pre_state = results[t][pre_state]
    tuple_result.insert(0, (obs[t - 1], crfgm.index2state[str(pre_state)]))
  return tuple_result

class CharDictInfo(object):
  def __init__(self):
    self.author=""
    self.create_date=""
    self.description=""

  def reset_info(self):
    self.author=""
    self.create_date=""
    self.description=""

  def show_info(self):
    print("\tAuthor:",self.author)
    print("\tCreate Date:",self.create_date)
    print("\tDesription:",self.description)

class CharDictMgr(object):

  def __init__(self):
    self.data={}
    self.info=CharDictInfo()

  def reset_dict(self):
    print("clear dictionary and info")
    self.data={}
    self.info.reset_info()

  def show_info(self):
    self.info.show_info()
    print("Total Nodes:{}".format(len(self.data.keys())))

  def load_root_node(self):
    self.data["^"] = crfgm.CRFNode("^")
    self.data["$"] = crfgm.CRFNode("$")

  def build_dict(self, tuple_source):
    '''
    用一个以seperator作为分隔符的以分好词的字符串扩充（更新）字典信息
    :param tuple_source:List表示的短语列表，每个元素是一个tuple:(c,s),c表示字本身，s表示其位置状态信息
    :return:更新字典信息
    '''
    source = _hat_tuple_list(tuple_source) # 让source 包含头尾信息
    pre_c, pre_s = source[0]  # source[0] 一定是("^","S")
    pre_n = self.get_char_node(pre_c)
    pre_n.touch_as(pre_s)
    length=len(source)
    for i in range(1,length):
      post_c, post_s = source[i]
      post_n = self.get_char_node(post_c)
      post_n.touch_as(post_s)
      pre_n.modify_matrix(pre_s,post_s)
      cur_p = pre_n.connect_to(post_n)
      cur_p.touch_as(pre_s,post_s)
      pre_n, pre_s = post_n, post_s
      pass

  def build_dict_from_file(self, file_name, sep=" "):
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
    print("learning:%s" % file_name)
    f = open(file_name, "r", encoding="utf-8")
    try:
      while True:
        content = f.readline()
        if(len(content)) == 0:
          break
        text=content
        tuple_source = mark_char_status(text)
        self.build_dict(tuple_source)

    except UnicodeDecodeError:
      print("Can't decode file:%s" % file_name)
    except IOError:
      print("IOError:\n")
    finally:
      f.close()

  def get_char_node(self,char="^"):
    """
    确保能获取一个节点，如果没有则在字典里新建一个
    :param char:
    :return:
    """
    if char is None or len(char) != 1:
      char = "^"
    n = self.data.get(char)
    if n is None:
      n = crfgm.CRFNode(char)
      self.data[char] = n  # add new node to dict
    return n

  def segment(self,sentence):
    """
    给一个句子或短语进行分词
    :param sentence: 一个经过预处理没有标点符号全部由中文汉字组成的短语或句子
    :return: [...,[(c,s),(c,s),(c,s),(c,s)],...]
    """
    sentence = _hat_string(sentence)  # 给待分析的句子加上头尾标志.
                                      # 例如："我喜欢你"变成"^我喜欢你$"
    length = len(sentence)  # length>=2
    if length == 2:
      return ""
    results = [[(0,None),(0,None),(0,None),(1,None)]] # "^"的状态

    pre_n = self.get_char_node(sentence[0])  # 节点："^"
    cur_n = self.get_char_node(sentence[1])  # 节点："我"

    pre_p = pre_n.search_path_to(cur_n) # 路径 从 "^"->"我"

    for t in range(1,length-1):   # 从"我"开始分析。结尾标志"$"不分析
      results.append([])   # 针对一个字，增加一个list，形式是[(p,s),(p,s),(p,s),(p,s)]
      post_n = self.get_char_node(sentence[t+1]) # 获取节点t+1："喜"

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
    return _analysis_results(sentence,results)

  def segment_file(self,source_file,target_file=None,show_detail=False):
    """
    对一个utf文本文件进行分词，分词结果保存在另一个文件里
    :param source_file: 原文件名
    :param target_file: 目标文件名
    :param show_detail: 是否显示详细信息
    :return:
    """

    rf = open(source_file, "r")
    if target_file is None or target_file == "":
      source_path,base_file = os.path.split(source_file)
      target_file = os.path.join(source_path,"seg_"+base_file)
    wf = open(target_file, "w")
    max_empty_lines = 10
    empty_lines = 0
    try:
      while empty_lines < max_empty_lines:
        text = rf.readline().strip()
        if (len(text)) == 0:
          empty_lines += 1
          continue
        else:
          empty_lines = 0
        result = self.segment(text)
        result = de_mark_char_status(result)
        if show_detail:
          print("语料片段: %s\n分词结果: %s\n"%(text,result))
        wf.write(result)
    except UnicodeDecodeError as e:
      print("Can't decode file:%s\n%s"%(file_name,e))
    except IOError as ioe:
      print("IOError:\n",ioe)
    finally:
      rf.close()
      wf.close()
    if show_detail:
      print("Complete! Segmentation result is in: %s"%target_file)

  def saveDictToXml(self, file_name):
    """
    save dict to an xml file
    :param file_name:
    :return: None
    """
    dictEle = Element("dict")
    dictEle.set("description", self.info.description)
    dictEle.set("count", str(len(self.data)))
    dictEle.set("author", self.info.author)
    cur_time = str(datetime.datetime.now())
    dictEle.set("create_date", cur_time)

    for key in self.data:
      cur_n = self.data[key]    # 一个节点
      nodeEle = SubElement(dictEle, "node") # 建立一个xmlEle
      crfgm.node2ele(cur_n,nodeEle)  # 关联xmlEle

      prePathsEle = SubElement(nodeEle, "pre_paths")
      prePathsEle.set("count", str(len(cur_n.pre_paths)))
      for path_key in cur_n.pre_paths:
        prePathEle = SubElement(prePathsEle, "path")
        pre_path = cur_n.pre_paths[path_key]
        crfgm.path2ele(pre_path,prePathEle,is_post_path = False)

      postPathsEle = SubElement(nodeEle, "post_paths")
      postPathsEle.set("count", str(len(cur_n.post_paths)))
      for path_key in cur_n.post_paths:
        postPathEle = SubElement(postPathsEle, "path")
        post_path = cur_n.post_paths[path_key]
        crfgm.path2ele(post_path,postPathEle,is_post_path = True)
    tree = ElementTree(dictEle)
    try:
      tree.write(file_name, encoding="utf-8", xml_declaration="version = 1.0")
      print("dictionary successfully saved in: %s"%file_name)
    except Exception as e:
      print("error occurs when write to xml file",e)

  def loadDictFromXml(self,file_name):
    """
    从xml文件加载字典
    :param file_name:
    :return:
    """
    self.reset_dict() # reset dict, add root and end nodes
    print("loading dictionary from:",file_name)
    try:
      tree = ElementTree(file=file_name)
      dictEle = tree.getroot()
      self.info.author = dictEle.attrib["author"]
      self.info.description = dictEle.attrib["description"]
      self.info.create_date = dictEle.attrib["create_date"]
      for nodeEle in dictEle:  # child node
        n_key = nodeEle.get("key")
        cur_n = self.get_char_node(n_key)
        crfgm.ele2node(nodeEle, cur_n)
        for pathsEle in nodeEle:  # pre_paths and post_paths
          for pathEle in pathsEle:
            #tgt_node, tgt_path = None,None
            if pathsEle.tag == "pre_paths":
              tgt_node = self.get_char_node(pathEle.get("pre_node"))
              tgt_path = cur_n.connect_from(tgt_node)
              crfgm.ele2path(pathEle, tgt_path)
            elif pathsEle.tag == "post_paths":
              tgt_node = self.get_char_node(pathEle.get("post_node"))
              tgt_path = cur_n.connect_to(tgt_node)
              crfgm.ele2path(pathEle, tgt_path)
            else:
              pass
      print("dictionary successfully loaded.")
    except Exception as e:
      print(e)
      print("error in loading dict from: %s"%file_name)


def test_save_load_dict_to_xml(dict_mgr:CharDictMgr):
  learning_file = "/home/yeqiang/chinese_resources/small_sample.utf8"
  dict_file = "/home/yeqiang/chinese_resources/small_sample_dict.xml"
  dict_mgr.build_dict_from_file(learning_file,sep=" ")
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
    "喜欢 这样 喜欢 亡羊补牢 共产党 我们 国家 共产党 领导 共产党 领导 中国 中国 中国 " \
    "共产党 共产党 共产党 共产党 "
  train_tuple = mark_char_status(train_str, " ")
  dict_mgr.build_dict(train_tuple)
  dict_mgr.show_info()
  dict_mgr.info.author="YeQiang"
  dict_mgr.info.description="小规模测试xml结构字典"
  print("save dict to xml file")
  dict_mgr.saveDictToXml(dict_file)
  dict_mgr.reset_dict()
  print("loading dic from xml file...")
  dict_mgr.loadDictFromXml(dict_file)
  dict_mgr.show_info()

def test_segment(dict_mgr:CharDictMgr):
  target_string = "希腊的经济结构比较单一"
  result = dict_mgr.segment(target_string)
  print(result)
  result = de_mark_char_status(result)

  print(result)
  while True:
    target_string = input("type your sentence:")
    if len(target_string) <= 0:
      break
    result = dict_mgr.segment(target_string)
    print(result)
    result = de_mark_char_status(result)
    print(result)

  while True:
    c = input("type 2 nodes to search path:")
    if len(c) < 2:
      break
    node = dict_mgr.data[c[0]]
    node2 = dict_mgr.data[c[1]]
    node.show_detail()
    node2.show_detail()
    path2 = node.search_path_to(node2)
    if path2 is not None:
      path2.show_detail()
    else:
      print("Path not exist")

def test_segment_file(dict_mgr:CharDictMgr):
  test_file = "/home/yeqiang/chinese_resources/pku_test.utf8"
  dict_mgr.segment_file(test_file,show_detail=True)

def test_segment_files_in_dir(dict_mgr:CharDictMgr,dir_path):
  filetool.dir_operate(dir_path,dict_mgr.segment_file,show_detail=False)

def test():
  dict_mgr = CharDictMgr()
  dict_file = "/home/yeqiang/chinese_resources/cn_dict.xml"
  learning_path = "/home/yeqiang/chinese_resources/for_learning"
  dict_mgr.loadDictFromXml(dict_file)
  dict_mgr.show_info()
  #filetool.dir_operate(learning_path,dict_mgr.build_dict_from_file)
  #dict_mgr.saveDictToXml(dict_file)
  dict_mgr.segment_file("/home/yeqiang/chinese_resources/doc_1utf8",show_detail=True)



if __name__ == "__main__":
  test()