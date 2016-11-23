# 此文件的主要功能是检测一个txt文件常用的编码格式，
# 把一个txt文件以UTF-8的格式重新编码
# -*- coding: UTF-8 -*-
import os

def checkfilencode(filename):
  encodings = ['gbk','utf-8','ascii','utf-16','utf-32']
  for curcode in encodings:
    flag = True
    f = open(filename, "r", encoding=curcode)
    try:
      content = f.read()
    except UnicodeDecodeError:
      flag = False
    finally:
      f.close()
    if flag == True:
      return curcode
  return None

# 讲一个目录下所有文本文件转化为utf8格式
def convertToUtf8TxtFile(dir):
  try:
    fdlist = os.listdir(dir)
  except FileNotFoundError:
    print("Not found:%s"%(dir))
    return
  for fileordir in fdlist:
    fdpath = os.path.join(dir, fileordir)
    if os.path.isdir(fdpath):
      convertToUtf8TxtFile(fdpath)
    else:
      filename = fdpath
      wfilename = filename+".utf8"
      codetype = checkfilencode(filename)
      if codetype is None or codetype is "utf-8":
        continue
      f = open(filename, "r", encoding=codetype)
      wf = open(wfilename, "w")
      try:
        content = f.read()
        if codetype is not "utf-8":
          wf.write(content)
          print("converting %s to utf-8:%s"%(codetype,filename))
      except:
        print("error occured!")
      finally:
        f.close()
        wf.close()
# 删除目录及其子目录下所有.txt结尾的文件，慎用
def deletetxtfilein(dir):
  try:
    fdlist = os.listdir(dir)
  except FileNotFoundError:
    print("Not found:%s"%(dir))
    return
  for fileordir in fdlist:
    fdpath = os.path.join(dir, fileordir)
    if os.path.isdir(fdpath):
      deletetxtfilein(fdpath)
    else:
      filename = fdpath
      l = len(filename)
      if filename[l-4:l] == ".txt":
        os.remove(filename)
        print("deleting:%s"%(filename))




if __name__ == "__main__":
  dirpath="/home/yeqiang/answer/"
  #convertToUtf8TxtFile(dirpath)
  deletetxtfilein(dirpath)