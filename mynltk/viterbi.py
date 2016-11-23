'''
这是一个应用在动态规划领域的维特比算法及一个例子。具体网址可见：
https://zh.wikipedia.org/wiki/%E7%BB%B4%E7%89%B9%E6%AF%94%E7%AE%97%E6%B3%95
本代码对原网址的代码进行了适当修改，主要目的是为了更好的显示运算逻辑及结果，其原理与原作完全一致
test函数提供您多次输入observation给出status的示例
修改者：叶强 qqiangye@gmail.com
'''

# 状态的样本空间
states = ('Healthy', 'Fever')


# 观测的样本空间
observations = ('normal', 'cold', 'dizzy')


# 起始个状态概率
start_probability = {'Healthy': 0.6, 'Fever': 0.4}


# 状态转移概率
transition_probability = {
  'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
  'Fever': {'Healthy': 0.4, 'Fever': 0.6},
}


# 状态->观测的发散概率
emission_probability = {
  'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
  'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
}


import numpy as np

def display_result(observations,result_m):
  """
  较为友好清晰的显示结果
  :param result_m:
  :return:
  """
  print(format("Viterbi Result","=^59s"))
  head = format("obs"," ^10s")
  head += format("Infered status"," ^18s")
  for s in states:
    head += format(s," ^15s")
  print(head)
  print(format("", "-^59s"))

  for obs,result in zip(observations,result_m):
    item = format(obs," ^10s")
    _, infered_status = max(zip(result.values(), result.keys()))
    item += format(infered_status," ^18s")
    for s in states:
      item += format(result[s]," >12.8f")
      if infered_status == s:
        item += "(*)"
      else:
        item +="   "

    print(item)
  print(format("", "=^59s"))


def viterbi(obs, states, start_p, trans_p, emit_p):

  result_m = [] # 存放结果
  pre_p = {}    # 存放前一次状态的概率
  for s in states:  # 对于每一个状态
    pre_p[s] = start_p[s]*emit_p[s][obs[0]] # 把第一个观测节点对应的各状态值计算出来
  result_m.append(pre_p)

  for ob in obs[1:]:
    cur_p = {}  # 存放在当前观测下得到的各状态的概率
    max_p,state_max_p = max(zip(pre_p.values(),pre_p.keys())) #查找前一次计算结果中最大概率及其状态
    for s in states: # 对于每一个状态,计算由前一时刻最大p及对应状态转移至各状态，并由各状态表现为观测结果的概率
      cur_p[s] = max_p * trans_p[state_max_p][s] * emit_p[s][ob]
    result_m.append(cur_p)  # 将当前的各状态P值存入结果列表
    pre_p = cur_p # 将当前状态赋值给先前状态，准备下次计算

  return result_m


def example():
  result = viterbi(observations,
                 states,
                 start_probability,
                 transition_probability,
                 emission_probability)
  display_result(observations,result)
  while True:
    user_obs = input("Now give me your observation, I will infer the status\n"
                "Using 'N' for normal, 'C' for cold and 'D' for dizzy\n"
                "Input here('q' to exit):")

    if len(user_obs) ==0 or 'q' in user_obs or 'Q' in user_obs:
      break
    else:
      obs = []
      for o in user_obs:
        if o == 'N' or o == 'n':
          obs.append("normal")
        elif o == 'C' or o == 'c':
          obs.append("cold")
        elif o == 'D' or o == 'd':
          obs.append("dizzy")
        else:
          pass
      result = viterbi(obs,
                       states,
                       start_probability,
                       transition_probability,
                       emission_probability)
      display_result(obs,result)



if __name__ == "__main__":
  example()
