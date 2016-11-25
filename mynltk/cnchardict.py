# 字典类
import os
import sys
import pickle
import math
try:
  from xml.etree.cElementTree import Element, SubElement, ElementTree
except ImportError:
  from xml.etree.ElementTree import Element, SubElement, ElementTree
import datetime
from graphmodel import BaseNode, BasePath
from char_dfa import *


# 中文单字字典
class CnCharDict:

  # init or reset dictionary
  def initDict(self):
    self.dict = {"^": self.root, "$": self.end}
    self.author = ""
    self.create_date = ""
    self.description = "Chinese Character Dictionary"

  # 新建一个字典，字典内至少有两个元素：一个是根，一个是末
  def __init__(self):
    self.root = BaseNode("^")
    self.end = BaseNode("$")
    self.initDict()

    self.dict = {"^": self.root, "$": self.end}
    self.author = ""
    self.create_date = ""
    self.description = "Chinese Character Dictionary"

  # 打印字典
  def show_dic_detail(self):
    for key in self.dict:
      self.dict[key].show_detail()


  # 让字典读取一段纯粹的中文字符串，不包括任何标点符号
  # 将字符串s的信息加入到字典中，添加字符串内新字，新的路径
  def learn_cn_str(self, s):
    if len(s) <= 0:
      return
    #if len(s) == 3 or len(s) == 4:
    #  newnode = node(s)
    #  self.dict[s] = newnode  # 添加3字词和4字成语
    s_id = s
    # s_id = str(hash(s))             # 给字符串一个唯一的编号
    # print("learning chinese phrase:%s" % (s))
    s = "^"+s+"$"       #给原字符串头尾加冒
    cur_node = self.dict.get(s[0])  # 在字典内查找字符串的第一个字["^"]

    for i in range(1, len(s)):
      post_n = self.dict.get(s[i])
      if post_n is None:
        post_n = BaseNode(s[i])
        self.dict[s[i]] = post_n  # add new node to dict
      new_path = cur_node.connectTo(post_n)  # add new path to dict
      new_path.touch(s_id)   # touch this path
      cur_node = post_n

  # 学习字符串数组，数组内字符串是经过处理过的全中文
  def learn_cn_strList(self,str_list,times=1):
    for i in range(times):
      for str in str_list:
        self.learn_cn_str(str)

  # learn chinese char from a file.
  # 从一个文本文件学习构建字典，文件可以是中英文混杂，要求是utf-8格式文件
  # 如果不是utf-8格式将忽视
  # txttencodeconverter 可以批量将常见其他编码格式的文本转化为utf-8文本
  # times 为重复学习的次数
  def learn_from_file(self, file,times=1):
    file_name = file
    print("learning:%s" % file_name)
    f = open(file_name, "r", encoding="utf-8")
    try:
      content = f.read()
      # pp 内包含一个有限自动机，用来预处理文本：分开中文、英文、标点、数字到不同的数组里
      str_list = pre_segment(content)
      cn_list = []
      for ctype, text in str_list:  # 数组元素是一个二元tuple:(a,b).
        if ctype == "cn":
          cn_list.append(text)
      self.learn_cn_strList(cn_list,times)
    except UnicodeDecodeError:
      print("Can't decode file:%s" %file_name)
    finally:
      f.close()

  # 将组织好的中文语料存放进一个文件夹，学习整个文件夹内内容
  def learnFromDirectory(self,dir_name):
    try:
      fd_list = os.listdir(dir_name)
    except FileNotFoundError:
      print("Not found:%s" % (dir_name))
      return

    for file_or_dir in fd_list:
      fd_path = os.path.join(dir_name, file_or_dir)
      if os.path.isdir(fd_path): # 如果是一个目录，递归调用自身
        self.learnFromDirectory(fd_path)
      else: # 如果是一个文件
        self.learn_from_file(fd_path)
      #print("nodes in dict:%d" % (len(self.dict)))


  # 使用字典尝试分割中文字符串
  # 参数display表示，是否显示相关信息
  def segmentCnStr(self, sentence, show_detail=False):
    if len(sentence) <= 1:
      return sentence
    #self.learn_cn_str(sentence)  # 先学习一次，录入可能碰到的生字
    if show_detail is True:
      print(sentence)
    out_cn_list = []
    s = "^" + sentence + "$"
    nums = [0, 0, 0]  # 初始链接数
    qiang_indexs = [] # 存放qiang指数数组，长度为原始句子长度-1
    for i in range(1, len(s) - 2):
      for j in range(3):
        obj_path = self.dict[s[i - 1 + j]].search_path_to(self.dict[s[i + j]])
        if obj_path is not None:
          factor = math.sqrt(math.log(len(self.dict[s[i - 1 + j]].post_paths)) * math.log(len(self.dict[s[i + j]].pre_paths)))
          nums[j] = obj_path.count * obj_path.count / (obj_path.count + factor)
        else: # 没找到这个路径，也就是有不认识的字或没学过的连接
          nums[j] = 0.5
        if i == 1:  # 限定"^"与第一个词的链接是1
          nums[0] = 0.5
        if i == len(s) - 3:  # 限定最后一个字与"$"链接也是1
          nums[2] = 0.5
      value = int((nums[1] * nums[1] * 100) / (nums[0] * nums[2])) / 100.00
      qiang_indexs.append(value)
    if show_detail is True:
      print(qiang_indexs)  # 显示qiang指数
    split = [0]  # 存储分割的索引
    for i in range(len(qiang_indexs) - 1):
      left = qiang_indexs[i]
      right = qiang_indexs[i + 1]

      if (left < 0.5 and right >1) or left == 0 or left+right <0.5:
        if split[len(split) - 1] < i + 1:  # 不重复添加
          split.append(i + 1)
      elif (right==0) or (left > 1 and right < 0.5) or (right>0 and left/right>1000):
        split.append(i+2)
      if left < 0.01 or (right>0 and left/right < 0.001):
        if split[len(split) - 1] < i + 1:  # 不重复添加
          split.append(i + 1)


      #if (left < 0.8 and right > 1 )   or left < 0.1 or  (right>0 and left/right <0.005):
      #  if split[len(split) - 1] < i + 1:  # 不重复添加
      #    split.append(i + 1)

      #elif (left > 1 and right < 0.8) or right<0.1 or (right>0 and left/right > 200):
      #  split.append(i+2)


      # if (left >= 1.0 and right < 1) or right < 0.001 :
      #   split.append(i + 2)
      # elif (left < 1 and right >= 1.0) or (left < 0.001) :
      #   # 在 i 处分割
      #   if split[len(split) - 1] < i + 1:  # 不重复添加
      #     split.append(i + 1)
      # elif left + right < 0.8 or left < 0.4:  # left + right < 0.4:
      #   if split[len(split) - 1] < i + 1:  # 不重复添加
      #    split.append(i + 1)
      else:
        pass
    split.append(len(sentence))
    for i in range(len(split) - 1):
      out_cn_list.append(sentence[split[i]:split[i + 1]])
    if show_detail is True:
      print(out_cn_list)
      _ = input("press any key to continue...")
    return out_cn_list

  # 分割一个字符串，可以使中英数字混
  def segmentStr(self, sentence, show_detail=False):
    if len(sentence)<=1:
      return sentence
    source_list = pre_segment(sentence)
    out_list = []
    for source_type, text in source_list:
      if source_type == "cn":
        out_cn_list = self.segmentCnStr(text, show_detail)
        for cn in out_cn_list:
          out_list.append(cn)
      else:
        out_list.append(text)
    return out_list

  def segmentFile(self, source_file, result_file, seperator=" ", show_detail=False):
    rf = open(source_file,"r")
    wf = open(result_file,"w")
    try:
      content = rf.read()
      str_list = pre_segment(content)
      for str_type,text in str_list:
        if str_type == "cn":
          cn_result = self.segmentCnStr(text,show_detail=show_detail)
          cn_string = ""
          for s in cn_result:
            cn_string += (s + seperator)
          wf.write(cn_string)
        else:
          # 将非中文写入
          text += seperator
          wf.write(text)
    finally:
      rf.close()
      wf.close()
    print("Complete! Segmentation result is in: %s"%result_file)


  # 将学习到的字典保存至指定文件,可以使用更好的xml输出函数保存字典
  def saveDictToFile(self,file_name):
    sys.setrecursionlimit(50000)
    f = open(file_name, "wb")
    try:
      pickle.dump(self.dict, f)
    finally:
      f.close()

  def loadDictFromFile(self,file_name):
    f = open(file_name, "rb")
    try:
      self.dict = pickle.load(f)
    finally:
      f.close()
    pass

  def displayDicInfo(self):
    print("dictionary brief info:")
    print("\tauthor:%s"%self.author)
    print("\tcreate date:%s"%self.create_date)
    print("\tdescription:%s"%self.description)
    print("\t%d nodes in the dictionary"%(len(self.dict)))

  # save dict to an xml file
  def saveDictToXml(self, file_name):
    dictEle = Element("dict")
    dictEle.set("description", self.description)
    dictEle.set("count", str(len(self.dict)))
    dictEle.set("author", self.author)
    cur_time = str(datetime.datetime.now())
    dictEle.set("create_date", cur_time)

    for key in self.dict:
      cur_n = self.dict[key]
      nodeEle = SubElement(dictEle, "node")
      nodeEle.set("key", cur_n.content)
      # nodeEle.set("content", cur_n.content)

      preBasePathsEle = SubElement(nodeEle, "pre_paths")
      preBasePathsEle.set("count", str(len(cur_n.prepaths)))
      for pathkey in cur_n.prepaths:
        preBasePathEle = SubElement(preBasePathsEle, "path")
        path = cur_n.prepaths[pathkey]
        preBasePathEle.set("touched", str(path.touchtimes))
        preBasePathEle.set("node", path.pre_n.content)

      postBasePathsEle = SubElement(nodeEle, "post_paths")
      postBasePathsEle.set("count", str(len(cur_n.posobj_paths)))
      for pathkey in cur_n.posobj_paths:
        postBasePathEle = SubElement(postBasePathsEle, "path")
        path = cur_n.posobj_paths[pathkey]
        postBasePathEle.set("touched", str(path.touchtimes))
        postBasePathEle.set("node", path.post_n.content)

    tree = ElementTree(dictEle)
    try:
      tree.write(file_name, encoding="utf-8", xml_declaration="version = 1.0")
      print("dictionary successfully saved in: %s"%(file_name))
    except Exception:
      print("error occurs when write to xml file")

  def loadDictFromXml(self,file_name):
    try:
      self.initDict()  # reset dict, add root and end nodes
      tree = ElementTree(file=file_name)
      dictEle = tree.getroot()
      self.author = dictEle.attrib["author"]
      self.description = dictEle.attrib["description"]
      self.create_date = dictEle.attrib["createtime"]
      node_count_from_attrib = int(dictEle.attrib["count"])
      for nodeEle in dictEle:  # child node
        n_key = nodeEle.get("key")
        node = BaseNode(n_key)
        self.dict[n_key] = node
        for pathsEle in nodeEle:  # pre_paths and post_paths
          for pathEle in pathsEle:
            tkey = pathEle.get("node")
            trg_node = self.dict.get(tkey)
            if trg_node is None:
              trg_node = BaseNode(pathEle.attrib["node"])
              self.dict[tkey] = trg_node
            if pathsEle.tag == "pre_paths":
              path = node.connect_from(trg_node)
              path.count = int(pathEle.attrib["touched"])
            else:
              path = node.connect_to(trg_node)
              path.count = int(pathEle.attrib["touched"])
      node_count_cal = len(self.dict)
      if node_count_from_attrib == node_count_cal:
        print("dictionary successfully loaded.")
      else:
        print("count in xml file(%d) dismatch real node count(%d)"%
              node_count_from_attrib,node_count_cal)
      #self.displayDicInfo()
    except Exception as e:
      print("error in loading dict from: %s"%file_name)
      print(e.args)
    finally:
      pass


def test():
  dict_xml_path="/home/yeqiang/pku_training_dict.xml"
  source_path = "/home/yeqiang/pku_test.utf8"
  target_obj_path = "/home/yeqiang/test_result.utf8"
  learning_file = "/home/yeqiang/chinese_resources/pku_training.utf8"

  mydict = CnCharDict()
  print("learning from file...")
  mydict.learn_from_file(learning_file)
  print("saving dict to xml...")
  mydict.author="yeqiang"
  mydict.create_date = str(datetime.datetime.now())
  mydict.saveDictToXml(dictxmlpath)
  mydict.displayDicInfo()
  mydict = None
  target = "I wrote 你好 this function. 我写了这个函数。2016-11-11"
  print("segmenting string: %s"%target)
  print("result is:")
  result = mydict.segmentStr(target)
  print(result)
  print("detail of node:的")
  mydict.dict["的"].printBaseNode()
  print("testing segment file to file...")
  print("source file: %s"%sourcepath)
  mydict.segmentFile(sourcepath, targeobj_path,show_detail=True)

  print("clear dictionary")
  mydict.initDict()
  mydict.displayDicInfo()

  print("reload dictionary from: %s"%dictxmlpath)
  mydict.loadDictFromXml(dictxmlpath)
  mydict.displayDicInfo()

  print("test completed.")


if __name__ == "__main__":
  # test()
  dictxmlpath="/home/yeqiang/pku_training_dict.xml"
  sourcepath = "/home/yeqiang/pku_test.utf8"
  targeobj_path = "/home/yeqiang/test_result.utf8"
  mydict = CnCharDict()
  mydict.loadDictFromXml("/home/yeqiang/large_dict.xml")

  #mydict.learnFromDirectory("/home/yeqiang/chinese_resources/")
  #mydict.saveDictToXml("/home/yeqiang/large_dict.xml")

  #mydict.learn_from_file("/home/yeqiang/extra_dict.utf8",1000)
  #mydict.segmentFile(sourcepath,targeobj_path,show_detail=True)
  while True:
    source = input("input text here:")
    if len(source) <= 1:
      break
    mydict.segmentStr(source,True)
