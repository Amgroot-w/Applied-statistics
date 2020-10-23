"""
《应用统计与凸优化》10.10作业：
（1）线性回归，拟合参数；
（2）参数检验。

学号： 2000900
姓名： 张敬川

调试记录：
1. 参数β3设置为-4时，无法通过最终的参数检验（p值为0.5），然而将其改为+4时，就可以通过检验？
bug原因：构造t检验统计量Tn时，没有加绝对值，导致Tn在参数β3处为负数，在t检验分布表上对应的数
值接近于1，输出的p-value始终为0.5，表示该参数未通过t检验。加上绝对值后，结果正常，β3能够正
常的通过t检验！

"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 多元线性回归：参数估计函数
def linear_regression(x, y):
    """
    :param x: (m,n)矩阵，决策变量的值
    :param y: (m,1)矩阵，输出的值
    :return: res1为参数估计值矩阵，res2为y的估计值矩阵，res3为残差矩阵
    """
    # 参数估计（最小二乘法）
    L = np.dot(x.T, x)      # 系数矩阵L
    C = np.linalg.inv(L)    # 相关矩阵C
    S = np.dot(x.T, y)      # 常数项矩阵S

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

    # 设置参数
    n = 100    # 样本个数
    p = 10     # 变量个数
    rho = 0.5  # 协方差矩阵的参数ρ
    beta = np.array([2, 3, 5, -4, 0, 0, 0, 0, 0, 0, 0]).reshape([-1, 1])  # 多元线性模型参数

    # 生成所需数据
    mean = np.zeros(p)  # 多元正态分布的均值
    cov = np.array([[rho**np.abs(i-j) for i in range(p)] for j in range(p)])  # 多元正态分布的协方差矩阵
    data_x = np.random.multivariate_normal(mean, cov, n)  # 生成100条10维的多元正态分布数据
    data_x = np.column_stack((np.ones([n, 1]), data_x))  # 加上一列全1列，构成完整的样本数据
    epsilon = np.random.standard_normal([100, 1])  # 生成100个标准正态分布值作为epsilon
    data_y = np.dot(data_x, beta) + epsilon  # 计算得到100个y的真实值

    # 参数估计&假设检验
    beta_h, y_h, resi, p_val = linear_regression(data_x, data_y)
    print("β的参数估计结果为：", beta_h.T, '\n')
    print("y的估计值Xβ为：", y_h.T, '\n')
    print("残差Y-Xβ为：", resi.T, '\n')
    print("H0: β1=0, H1: β1≠0\t\t检验结果：p-value为%.6f\t即：拒绝原假设H0，认为β1≠0" % p_val[1])
    print("H0: β4=0, H1: β4≠0\t\t检验结果：p-value为%.6f\t即：接受原假设H0，认为β4=0" % p_val[4])

    # 可视化
    plt.figure()  # 对比图
    plt.bar(range(n), data_y[:, 0], label='y的真实值')
    plt.bar(range(n), y_h[:, 0], label='y的估计值')
    plt.xlabel('样本')
    plt.ylabel('y值')
    plt.legend(loc='upper right')
    plt.show()
    plt.figure()  # 残差图
    plt.scatter(range(n), resi, c='orange', marker='o', s=120, alpha=0.7, linewidths=1, edgecolors='k')
    plt.plot(range(n), resi, '--k', alpha=0.6)
    plt.plot([-5, 105], [0, 0], '--k', alpha=0.7)
    plt.xlabel('样本')
    plt.ylabel('残差值')
    plt.xlim(-5, 105)
    plt.ylim(-3.5, 3.5)
    plt.show()





