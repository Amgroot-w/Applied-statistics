"""
《应用统计与凸优化》大作业：
(1) 使用附件“dataset.csv”中的数据，建立X1-X11，与Y之间的模型。
(2) 可以使用正则化方法或者变量选择的方法对模型中的变量进行筛选，结果列出每个模型的表达形式及残差平方和。

学号：2000900    姓名：张敬川    2020.11.17

运行要求：
项目文件夹下须有function.py和cap.py两个文件，以及数据dataset.csv

调试记录：
1. LASSO回归，参数怎么才会变为0? 按照坐标下降法，参数不太可能会自动训练为0啊?
”在变量的变化程度很小时“，将该维度参数设为0，我用pk作为判断依据，当pk位于（-λ，λ）中时，
表示变化程度“很小”，那么此时将参数设为0。实际调试发现，pk这个数太大了，都是几千几百，不
可能小到设置的阈值范围中，所以该方法不可行！

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from function import LR, LR_sklearn, bp

# %% 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# %% 导入数据
data = pd.read_csv('dataset.csv')
# 选择决策变量
data_x = data.iloc[:, 1:12].values  # 选择全部决策变量
# data_x = data[['X1', 'X2', 'X3', 'X6', 'X11']].values  # 剔除无关的变量
# 选择因变量
data_y = data.iloc[:, 12].values.reshape(-1, 1)


# %% 模型拟合
beta_h1, y_h1, p_val1 = LR(data_x, data_y, reg='None')  # 无正则化

beta_h2, y_h2, p_val2 = LR(data_x, data_y, reg='Ridge', lamda=0.1)  # L2正则化

beta_h3, y_h3, _ = LR(data_x, data_y, reg='LASSO', lamda=0.1, alpha=0.1, epochs=2000)  # L1正则化

beta_h4, y_h4 = LR_sklearn(data_x, data_y, alpha=0.1)  # sklearn库实现LASSO回归

beta_h_bp, y_h_bp = bp(data_x, data_y, epochs=5000, alpha=0.5, lamda=0.01)  # BP神经网络














