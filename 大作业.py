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
from function import LR

# %% 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# %% 导入数据
data = pd.read_csv('dataset.csv')
data_x = data.iloc[:, 1:12].values
data_y = data.iloc[:, 12].values.reshape(-1, 1)


# %% 线性回归
beta_h, y_h, resi, p_val = LR(data_x, data_y)




