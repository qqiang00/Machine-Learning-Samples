# this py includes functions about pre process for NLP problems
# 可以初步分割字符串内的中文英文和数字。中文需要进一步分词
# 英文和数字已经效果还不错。在数字的分式接百分号这一点没有做好
# 具体机制参考状态转换表
# author：yeqiang
# create date:2016-11-08

# 状态数组，对应status_transit_table内的行
dfa_status =["start", "1", "2", "3", "4", "5", "6", "7",
             "cn", "en", "num", "other"]
# 字符类别数组，对应status_transit_table内的列
type_char = ["cn", "en", "num", ".", "-", "'", "/", "%","other"]

# 状态转移表，确定了某个状态在一个特定字符输入后的新状态，
# 其中8-11为结束状态，1-6为中间状态
status_transit_table =\
  [
    #cn  en  num  .   -   '   /   %, other
    [1,  2,   3,  7,  4,  4,  4,  4,  4],   # start
    [1,  8,   8,  8,  8,  8,  8,  8,  8],   # 1
    [9,  2,   9,  9,  2,  2,  9,  9,  9],   # 2
    [10, 10,  3,  5, 10, 10,  5,  6, 10],   # 3
    [11, 11, 11, 11,  4,  4,  4,  4,  4],   # 4
    [10, 10,  5, 10, 10, 10, 10, 10, 10],   # 5
    [10, 10, 10, 10, 10, 10, 10, 10, 10],   # 6
    [11, 11,  5,  4, 4,   4,  4,  4,  4]    # 7
  ]

import re
# regular expression of different type of c
cn_words = re.compile(u"([\u4e00-\u9fff])")
en_words = re.compile(u"([A-Za-z])")
num = re.compile(u"([0-9])")
p_dot = re.compile(u"\.")
p_link = re.compile(u"-")
p_quote = re.compile(u"'")
p_split = re.compile(u"/")
p_per = re.compile(u"%")

# 获取字符c在状态转化表里的类别，返回的是该类别在字符类别数组内的索引
def typeof(c):
  if cn_words.search(c, 0) is not None:
    return 0
  elif en_words.search(c, 0) is not None:
    return 1
  elif num.search(c, 0) is not None:
    return 2
  elif p_dot.search(c, 0) is not None:
    return 3
  elif p_link.search(c, 0) is not None:
    return 4
  elif p_quote.search(c, 0) is not None:
    return 5
  elif p_split.search(c, 0) is not None:
    return 6
  elif p_per.search(c, 0) is not None:
    return 7
  else:
    return 8

# 朱字符测试类别函数的正确性
def test_type(s):
  for i in range(len(s)):
    print("%s:%7s"%(s[i], type_char[typeof(s[i])]))

# 讲一段文字初步分解为中文、英文、数字以及其他不明确符号
# 不支持网址分析
def pre_segment(s):
  # 为方便处理，字符串结尾添加一个空字符
  s += " "
  start_index = 0 # 初始索引为：0
  cur_status = 0  # 初始状态为start：0
  results = []    # 存放解析结果数组
  for i in range(len(s)):
    c_type = typeof(s[i])
    cur_status = status_transit_table[cur_status][c_type]
    if 8 <= cur_status <=11:
      # 分割成功
      results.append((dfa_status[cur_status],s[start_index:i]))
      # 新分割的开始
      start_index = i
      # 新的当前状态已经不是Start,而是输入了索引为i的字符
      cur_status = status_transit_table[0][c_type]
  # 字符串结尾为空字符，导致便利结束后当前状态只可能在4：其他处.
  if start_index < i:
    results.append((dfa_status[11],s[start_index:i]))
  return results

def test():
  s = "我是中国人，中国有14.6亿人口，其中90%以上是汉族，还有大约1/10是回族。 " \
      "We love China. She's a great country! " \
      "my-great-china, 我们Love您！ "
  # s = '130-4 0.123 .123 12/45% 12% .12% %213 http://www.sina.com.cn. '
  # s = "You are my teacher. I'm your student.     "
  # s= " "
  results = pre_segment(s)
  for i in range(len(results)):
    print(results[i])

if __name__ == "__main__":
  test()