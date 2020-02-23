# -*- coding: utf-8 -*-

# 上海股市指数走势预测，使用时间序列模型ARMA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA  #时间序列
import warnings
from itertools import product
from datetime import datetime
warnings.filterwarnings('ignore')
# 数据加载
df = pd.read_csv('.\\shanghai_1990-12-19_to_2019-2-28.csv')
# 将时间作为df的索引
df.Timestamp = pd.to_datetime(df.Timestamp) #time 的series str格式改为时间戳类型，
df.index = df.Timestamp
# 数据探索
print(df.head())
# 按照月，季度，年来统计
df_month = df.resample('M').mean()
df_Q = df.resample('Q-DEC').mean()
df_year = df.resample('A-DEC').mean()
# 按照天，月，季度，年来显示比特币的走势
fig = plt.figure(figsize=[15, 7])
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.suptitle('上海股市指数', fontsize=20)
plt.subplot(221)
plt.plot(df.Price, '-', label='按天')
plt.legend()
plt.subplot(222)
plt.plot(df_month.Price, '-', label='按月')
plt.legend()
plt.subplot(223)
plt.plot(df_Q.Price, '-', label='按季度')
plt.legend()
plt.subplot(224)
plt.plot(df_year.Price, '-', label='按年')
plt.legend()
plt.show()
# 设置参数范围
ps = range(0, 4)
qs = range(0, 4)
parameters = product(ps, qs)  #笛卡儿积得到6组数据对，元组类型
parameters_list = list(parameters)
# 寻找最优ARMA模型参数，即best_aic最小
results = []
best_aic = float("inf") # 正无穷
for param in parameters_list:
    try:
        model = ARMA(df_month.Price,order=(param[0], param[1])).fit()
    except ValueError:
        print('参数错误:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])
# 输出最优模型
result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic'] #赋值属性的方式生成属性
print('最优模型: ', best_model.summary())
# 比特币预测
df_month2 = df_month[['Price']] #只取价格列
# print(df_month2.shape)
# import sys
# sys.exit()
date_list = [datetime(2019, 3, 31),datetime(2019, 4, 30), datetime(2019, 5, 31), datetime(2019, 6, 30),datetime(2019,7,31),\
             datetime(2019,8,31),datetime(2019,9,30),datetime(2019,10,31),datetime(2019,11,30),datetime(2019,12,31),datetime(2020,1,31),datetime(2020,2,29)]
future = pd.DataFrame(index=date_list, columns= df_month.columns)
df_month2 = pd.concat([df_month2, future]) #axis = 0 纵向堆叠
df_month2['forecast'] = best_model.predict(start=0, end=351) #predict(start, end) 函数进行预测，其中 start 为预测的起始时间，end 为预测的终止时间。
#df_month-1.shape(339,1)加预测10个月，即349
print(df_month2['forecast'][338:])
# import sys
# sys.exit()

# 上海股市价格预测结果显示
plt.figure(figsize=(20,7))
df_month2.Price.plot(label='实际指数')
df_month2.forecast.plot(color='r', ls='--', label='预测指数')
plt.legend()
plt.title('上海股市价格（月）')
plt.xlabel('时间')
plt.ylabel('指数')
plt.show()
# @Time : 2020/2/23 9:00
# @Author : pingguo
# @File : 41shanghai.py