# -*- coding: utf-8 -*-
# @File:的膜.py
# @Author:south wind
# @Date:2022-12-01
# @IDE:PyCharm
def nearestValidPoint(x: int, y: int, points):
    output = -1
    res = -1
    for i, (a, b) in enumerate(points):
        if a == x or b == y:
            if output > (abs(a - x) + abs(b - y)):
                res = i
    return res
x=nearestValidPoint(3,4,[[1,2],[3,1],[2,4],[2,3],[4,4]])
print(x)