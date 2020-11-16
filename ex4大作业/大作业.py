"""
《应用统计与凸优化》大作业：
(1) 使用附件“dataset.csv”中的数据，建立X1-X11，与Y之间的模型。
(2) 可以使用正则化方法或者变量选择的方法对模型中的变量进行筛选，结果列出每个模型的表达形式及残差平方和。

学号： 2000900
姓名： 张敬川
2020.11.12

调试记录：

"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from function import LR, LR_L1, bp

# %% 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# %% 导入数据
data = pd.read_csv('dataset.csv')
data_x = data.iloc[:, 1:12].values
# data_x = data[['X1', 'X2', 'X3', 'X6']]
data_y = data.iloc[:, 12].values.reshape(-1, 1)


# %% 线性回归
beta_h1, y_h1, p_val1 = LR(data_x, data_y, lamda=0)  # 线性回归（无正则化）

beta_h2, y_h2, p_val2 = LR(data_x, data_y, lamda=1)  # 线性回归（L2正则化）

beta_h3, y_h3 = LR_L1(data_x, data_y, epochs=2, alpha=0.07, lamda=1)  # 线性回归（L1正则化）

# y_hat4, w1, b1 = bp(data_x, data_y, epochs=5000, alpha=0.5, lamda=0.01)  # bp神经网络

