"""
function.py

"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import cap

# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 残差图
def resi_plot(m, resi):

    plt.figure()
    plt.scatter(range(m), resi, c='orange', marker='o', s=120, alpha=0.7, linewidths=1, edgecolors='k')
    plt.plot(range(m), resi, '--k', alpha=0.6)
    plt.plot([-5, 105], [0, 0], '--k', alpha=0.7)
    plt.xlabel('样本')
    plt.ylabel('残差值')
    plt.xlim(-5, 105)
    plt.ylim(-3.5, 3.5)
    plt.show()

# 残差平方和
def resi_sum(resi):
    res = sum(resi**2)
    print('残差平方和：%.4f' % res)

# 多元线性回归：无正则化 + L2正则化
def LR(x, y, lamda=0):
    """
    :param x: (m,n)矩阵，决策变量的值
    :param y: (m,1)矩阵，输出的值
    :param lamda: 正则化参数
    :return: beta_hat为参数估计值矩阵，y_hat为y的估计值矩阵，
             residual为残差矩阵，p_value为t检验的p值
    """
    # 预处理
    m, n = x.shape  # 样本个数n，变量个数p
    x = np.column_stack((np.ones([m, 1]), x))  # 加上全1列作为x0（截距项）

    # 参数估计（最小二乘法）
    Id = np.identity(n+1)              # 生成单位阵
    Id[0, 0] = 0                       # theta0不参与正则化
    L = np.dot(x.T, x) + 1/2*lamda*Id  # 系数矩阵L
    C = np.linalg.inv(L)               # 相关矩阵C
    S = np.dot(x.T, y)                 # 常数项矩阵S
    beta_hat = np.dot(C, S)            # 参数估计值矩阵β_hat
    y_hat = np.dot(x, beta_hat)        # y的估计值y_hat
    residual = y - y_hat               # 残差residual

    # 假设检验（t检验）
    sigma_square = np.sum(residual**2) / (m-n-1)         # 求σ_hat^2的无偏估计值
    gamma = np.diagonal(C).reshape(-1, 1)                # 取出相关矩阵C的对角线元素赋值给gamma
    Tn = beta_hat / np.sqrt(sigma_square * gamma)        # t检验统计量
    p_value = (1 - stats.t.cdf(np.abs(Tn), m-n-1)) / 2   # t检验的p—value

    resi_plot(m, residual)  # 残差图
    resi_sum(residual)  # 残差平方和

    return beta_hat, y_hat, p_value

# 多元线性回归：L1正则化
def LR_L1(x, y, epochs, alpha, lamda):
    x = np.column_stack((np.ones([x.shape[0], 1]), x))  # 加上全1列作为x0（截距项）
    m = x.shape[0]  # 样本数
    n = x.shape[1]  # 特征数
    theta = np.random.uniform(-1, 1, [x.shape[1], 1])  # 参数初始化
    delta = np.zeros([n, 1])  # 梯度初始化
    cost_history = {'epoch': [], 'cost': []}  # 字典记录误差变化
    # 训练
    for epoch in range(epochs):
        # 假设函数h(θ)
        h = np.matmul(x, theta)
        # 均方误差损失 + L1正则化项
        J = cap.mse(h, y) + lamda*np.linalg.norm(theta[1:n, :], 1)
        # 坐标下降法！！！（梯度下降法不再适用！）
        delta[0, :] = 1/m * np.matmul(x.T[0, :], h-y)  # theta0不加正则化
        delta[1:n, :] = 1/m * np.matmul(x.T[1:n, :], h-y) + lamda*np.sign(theta[1:n, :])
        # 参数更新
        theta = theta - alpha * delta
        if
        # 记录误差cost
        cost_history['epoch'].append(epoch)
        cost_history['cost'].append(J)

    beta_hat = theta  # 参数估计值
    y_hat = np.matmul(x, beta_hat)  # y的估计值
    residual = y_hat - y  # 残差

    resi_plot(m, residual)  # 残差图
    resi_sum(residual)  # 残差平方和

    return theta, y_hat

# bp神经网络
def bp(x, y, epochs, alpha, lamda):
    # 超参数
    train_num = x.shape[0]  # 样本数
    input_num = x.shape[1]  # 输入节点数
    hidden_num = 8  # 隐层节点数
    output_num = 1  # 输出节点数

    # 初始化权重
    w1 = np.random.uniform(-0.5, 0.5, [input_num, hidden_num])
    w2 = np.random.uniform(-0.5, 0.5, [hidden_num, output_num])
    b1 = np.zeros(hidden_num)
    b2 = np.zeros(output_num)

    # 训练
    cost = []
    for epoch in range(epochs):
        # 前向传播
        hidden_in = np.dot(x, w1) + b1
        hidden_out = cap.sigmoid(hidden_in)
        network_in = np.dot(hidden_out, w2) + b2
        network_out = network_in

        # 记录总误差
        J = cap.mse(network_out, y) + 1/(2*train_num) * lamda * (np.sum(w2**2) + np.sum(w1**2))
        cost.append(J)
        # 反向传播
        output_delta = network_out - y

        hidden_delta = np.multiply(np.dot(output_delta, w2.T),
                                   np.multiply(hidden_out, 1-hidden_out))
        # 梯度更新
        dw2 = 1/train_num * (np.dot(hidden_out.T, output_delta) + lamda*w2)
        db2 = 1/train_num * np.dot(np.ones([train_num, 1]).T, output_delta)
        dw1 = 1/train_num * (np.dot(x.T, hidden_delta) + lamda*w1)
        db1 = 1/train_num * np.dot(np.ones([train_num, 1]).T, hidden_delta)
        w2 = w2 - alpha*dw2
        w1 = w1 - alpha*dw1
        b2 = b2 - alpha*db2
        b1 = b1 - alpha*db1

    y_hat = network_out  # y的估计值
    residual = output_delta  # 残差

    resi_plot(x.shape[0], residual)  # 残差图
    resi_sum(residual)  # 残差平方和

    return y_hat, w1, b1


















