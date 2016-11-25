#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 24/11/16 6:58 PM
# @Author  : YeQiang
# @Email   : qqiangye@gmail.com   
# @Site    : 
# @File    : strtool.py.py

'''
TODO: 添加对这段代码的描述
'''
import numpy as np

def list2string(f_list,sep=" "):
  """
  给出一个float一维数据数组，返回拥有该数组的字符串形式，用
  :param f_list:
  :param sep: 分隔符
  :return:
  """
  result = ""
  for i in range(len(f_list)):
    result+=str(int(f_list[i]))
    if i< len(f_list)-1:
      result += sep
  return result

def string2list(source,sep=" "):
  """
  将一个字符串转换为数组
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
    result += str(int(matrix[i][0]))
    if i < size-1:
      result += sep
  return result

def string2matrix(source, sep=" ",shape=(4,4)):
  l = string2list(source)
  matrix = np.asarray(l,dtype=float)
  matrix.shape=shape
  return matrix