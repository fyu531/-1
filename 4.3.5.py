import numpy as np
import json

# Viterbi算法求测试集最优状态序列
def Viterbi(sentence, array_pi, array_a, array_b, STATES):
    weight = [{}]  # 动态规划表：weight[t][state]表示t时刻到达state的最大概率
    path = {}

    # 处理第一个字符（未登录词赋予极小概率）
    for state in STATES:
        if sentence[0] in array_b[state]:
            b_prob = array_b[state][sentence[0]]
        else:
            b_prob = -3.14e+100  # 未登录词的观测概率设为极小值
        weight[0][state] = array_pi[state] + b_prob
        path[state] = [state]  # 记录当前状态的路径

    # 迭代处理后续字符
    for i in range(1, len(sentence)):
        weight.append({})
        new_path = {}
        current_char = sentence[i]

        for state0 in STATES:  # state0：当前字符的状态（t时刻）
            items = []
            # 计算当前状态state0的所有可能前驱状态state1（t-1时刻）的概率
            for state1 in STATES:  # state1：前一个字符的状态
                # 处理未登录词的观测概率
                if current_char in array_b[state0]:
                    b_prob = array_b[state0][current_char]
                else:
                    b_prob = -3.14e+100
                # 概率公式：前一状态概率 + 状态转移概率 + 观测概率（对数域，加法替代乘法）
                prob = weight[i-1][state1] + array_a[state1][state0] + b_prob
                items.append((prob, state1))  # 存储（概率，前驱状态）

            # 选择最大概率对应的前驱状态
            best_prob, best_state = max(items)
            weight[i][state0] = best_prob
            new_path[state0] = path[best_state] + [state0]  # 更新路径

        path = new_path  # 替换为当前时刻的所有路径

    # 选择最后一个字符的最优状态（优先E/S，因句子结尾只能是E或S）
    final_probs = [(weight[-1][state], state) for state in STATES if state in ['E', 'S']]
    if not final_probs:  # 极端情况：无E/S状态，取所有状态的最大值
        final_probs = [(weight[-1][state], state) for state in STATES]
    best_prob, best_state = max(final_probs)
    return path[best_state]


# 根据状态序列进行分词
def tag_seg(sentence, tag):
    word_list = []
    start = 0
    if len(tag) != len(sentence):
        return ["分词失败：状态序列与句子长度不匹配"]
    
    for i in range(len(tag)):
        if tag[i] == 'S':  # 单字成词
            word_list.append(sentence[i])
            start = i + 1
        elif tag[i] == 'B':  # 词的开始
            start = i
        elif tag[i] == 'E':  # 词的结束
            word_list.append(sentence[start:i+1])
            start = i + 1
        elif tag[i] == 'M':  # 词的中间，跳过
            continue
    
    return word_list


if __name__ == '__main__':
    # 加载HMM模型参数（由训练集统计得到）
    pramater = json.load(open('4.3.5/hmm_states.txt', encoding='utf-8'))
    array_A = pramater['states_matrix']   # 状态转移概率矩阵（B/M/E/S之间的转移概率）
    array_B = pramater['observation_matrix']   # 观测概率矩阵（状态→字符的概率）
    array_Pi = pramater['init_states']  # 初始状态概率分布（句子第一个字符的状态概率）
    STATES = ['B', 'M', 'E', 'S']  # HMM状态：B(词首)、M(词中)、E(词尾)、S(单字)

    # 测试句子
    test = "中国游泳队在东京奥运会上取得了优异的成绩"
    # 1. Viterbi算法得到最优状态序列
    tag = Viterbi(test, array_Pi, array_A, array_B, STATES)
    print("最优状态序列：", tag)
    # 2. 根据状态序列分词
    seg_result = tag_seg(test, tag)
    print("分词结果：", '/ '.join(seg_result))
