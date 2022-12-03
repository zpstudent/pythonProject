# 获取股票信息
import pandas_datareader as web
import matplotlib.pyplot as plt
stock_code = list(input('请输入要查询的股票代码：（以逗号隔开）').split(','))

n = len(stock_code)
for i in range(n):
    stock_code[i] += '.SS'
    data = web.get_data_yahoo(str(stock_code[i]))
    print(f'---------------------------{stock_code[i]}---------------------------')
    print(data.head(5))
    print('\n')
