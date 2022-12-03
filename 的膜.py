# -*- coding: utf-8 -*-
# @File:的膜.py
# @Author:south wind
# @Date:2022-12-01
# @IDE:PyCharm
def minOperations(boxes: str):
    initial_state = []
    n = len(boxes)
    for i in range(n):
        if boxes[i] == '1':
            num_i = [0] * n
            for j in range(n):
                num_i[j] = abs(j - i)
            initial_state.append(num_i)
    print(initial_state)
    return [sum(initial_state[i][:]) for i in range(n)]
print(minOperations('110'))