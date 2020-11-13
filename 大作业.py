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
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 多元线性回归
def linear_regression(x, y):
    """
    :param x: (m,n)矩阵，决策变量的值
    :param y: (m,1)矩阵，输出的值
    :return: beta_hat为参数估计值矩阵，y_hat为y的估计值矩阵，residual为残差矩阵
    """

    # 预处理
    n, p = x.shape  # 样本个数n，变量个数p
    x = np.column_stack((np.ones([n, 1]), x))  # 加上全1列作为x0（截距项）

    # 参数估计（最小二乘法）
    L = np.dot(x.T, x)           # 系数矩阵L
    C = np.linalg.inv(L)         # 相关矩阵C
    S = np.dot(x.T, y)           # 常数项矩阵S
    beta_hat = np.dot(C, S)      # 参数估计值矩阵β_hat
    y_hat = np.dot(x, beta_hat)  # y的估计值y_hat
    residual = y - y_hat         # 残差residual

    # 假设检验（t检验）
    sigma_square = np.sum(residual**2) / (n-p-1)         # 求σ_hat^2的无偏估计值
    gamma = np.diagonal(C).reshape(-1, 1)                # 取出相关矩阵C的对角线元素赋值给gamma
    Tn = beta_hat / np.sqrt(sigma_square * gamma)        # t检验统计量
    p_value = (1 - stats.t.cdf(np.abs(Tn), n-p-1)) / 2   # t检验的p—value

    return beta_hat, y_hat, residual, p_value


if __name__ == "__main__":

    # 导入数据
    data = pd.read_csv('dataset.csv')
    data_x = data.iloc[:, 1:12].values
    data_y = data.iloc[:, 12].values.reshape(-1, 1)

    # 参数估计&假设检验
    beta_h, y_h, resi, p_val = linear_regression(data_x, data_y)

    # 可视化
    plt.figure()  # 残差图
    plt.scatter(range(data_x.shape[0]), resi, c='orange', marker='o', s=120, alpha=0.7, linewidths=1, edgecolors='k')
    plt.plot(range(data_x.shape[0]), resi, '--k', alpha=0.6)
    plt.plot([-5, 105], [0, 0], '--k', alpha=0.7)
    plt.xlabel('样本')
    plt.ylabel('残差值')
    plt.xlim(-5, 105)
    plt.ylim(-3.5, 3.5)
    plt.show()




