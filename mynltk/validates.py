'''
这个文件是一个验证文件，内部留了一些网上下载或自己写的一些验证想法的函数
和包内其他文件关系不大
'''
#!coding:utf-8

import re
import sys

# FIXME: 突出颜色显示的注释
# 测试匹配中文信息
def TestReChinese():
  source = "        数据结构模版----单链表SimpleLinkList[带头结点&&面向对象设计思想](C语言实现)"
  source.encode()
  temp = source # source.decode('utf8')
  print("同时匹配中文英文")
  print("--------------------------")
  xx = u"([\w\W\u4e00-\u9fff]+)"
  pattern = re.compile(xx)
  results = pattern.findall(temp)
  for result in results:
    print(result)
  print("--------------------------")
  print("只匹配中文")
  print("--------------------------")
  xx = u"([\u4e00-\u9fff]+)"
  pattern = re.compile(xx,0)
  results = pattern.findall(temp)

  for result in results:
    print(result)
  print("--------------------------")




# 这是一个测试函数，完整的函数可以在cnchardict.py里找到
try:
  from xml.etree.cElementTree import Element, SubElement, ElementTree
except ImportError:
  from xml.etree.ElementTree import Element, SubElement, ElementTree

#from cnchargraph import Node, Path
import datetime


def saveDictToXml(dict, filepath):

  dictEle = Element("dict")
  dictEle.set("description","Chinese Character Dictionary")
  dictEle.set("count",str(len(dict)))
  dictEle.set("author", "yeqiang")
  cur_time = str(datetime.datetime.now())
  dictEle.set("createtime",cur_time)

  for key in dict:
    cur_n = dict[key]
    print(cur_n)
    nodeEle = SubElement(dictEle, "node")
    nodeEle.set("key", cur_n.content)
    #nodeEle.set("content", cur_n.content)

    prePathsEle = SubElement(nodeEle, "pre_paths")
    prePathsEle.set("count", str(len(cur_n.prepaths)))
    for pathkey in cur_n.prepaths:
      prePathEle = SubElement(prePathsEle, "path")
      path = cur_n.prepaths[pathkey]
      #prePathEle.set("key",str(path.pid))
      prePathEle.set("touched", str(path.touchtimes))
      prePathEle.set("node", path.pre_n.content)
      #prePathEle.text = path.pid

    postPathsEle = SubElement(nodeEle, "post_paths")
    postPathsEle.set("count", str(len(cur_n.postpaths)))
    for pathkey in cur_n.postpaths:
      postPathEle = SubElement(postPathsEle, "path")
      path = cur_n.postpaths[pathkey]
      #postPathEle.set("key", str(path.pid))
      postPathEle.set("touched", str(path.touchtimes))
      postPathEle.set("node", path.post_n.content)

  tree = ElementTree(dictEle)
  tree.write(filepath, encoding="utf-8", xml_declaration="version = 1.0")


if __name__ == "__main__":
  # 测试正则表达式

  print(sys.getdefaultencoding())


  TestReChinese()