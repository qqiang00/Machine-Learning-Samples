#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 24/11/16 8:18 PM
# @Author  : YeQiang
# @Email   : qqiangye@gmail.com   
# @Site    : 
# @File    : filetool.py.py

import os
import sys

def dir_operate(dir_name, operate,*args,**kwargs):
  """
  递归对一个文件夹内的每一个文件执行一个操作
  :param dir_name:目录名
  :param operate: 执行的操作，操作的对象是文件名。比如打开一个文件等
  :return: None
  """
  try:
    fd_list = os.listdir(dir_name)
  except FileNotFoundError:
    print("Not found directory:%s" %dir_name)
    return
  for file_or_dir in fd_list:
    fd_path = os.path.join(dir_name, file_or_dir)
    if os.path.isdir(fd_path):  # 如果是一个目录，递归调用自身
      dir_operate(fd_path,operate,*args,*kwargs)
    else:  # 如果是一个文件
      #print("processing:",fd_path)
      operate(fd_path,*args,**kwargs)


def file_process(file_path,show_detail=True):
  print("processing:",file_path)
  if show_detail is True:
    print("show processing detail")


def test():
  test_dir= "/home/yeqiang/chinese_resources/answer/C4-Literature"
  dir_operate(test_dir,file_process,show_detail=True)


if __name__ == "__main__":
  test()