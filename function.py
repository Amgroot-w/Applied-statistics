"""
function.py

"""
import numpy as np
from scipy import stats
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import cap

# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 残差图
def resi_plot(resi, reg):
    if reg == 'None回归':
        reg = '不加正则化'
    plt.figure()
    plt.scatter(range(resi.shape[0]), resi, c='orange', marker='o', s=120, alpha=0.7, linewidths=1, edgecolors='k')
    plt.plot(range(resi.shape[0]), resi, '--k', alpha=0.6)
    plt.plot([-5, 105], [0, 0], '--k', alpha=0.7)
    plt.xlabel('样本')
    plt.ylabel('残差值')
    plt.title('%s 残差图' % reg)
    plt.xlim(-5, 105)
    plt.ylim(-3.5, 3.5)
    plt.show()

# 残差平方和
def resi_sum(resi, reg):
    res = sum(resi**2)
    if reg == 'None回归':
        reg = '不加正则化'
    print('%s，残差平方和(RSS)：%.4f' % (reg, res))

# 多元线性回归
def LR(x, y, reg='None', lamda=0, alpha=0.01, epochs=2000):
    """
    :param x: (m,n)矩阵，决策变量的值
    :param y: (m,1)矩阵，输出的值
    :param reg: 选择正则化方式（L1、L2）
    :param lamda: 正则化参数
    :param alpha: 训练的学习率
    :param epochs: 迭代次数

    :return: beta_hat为参数估计值矩阵，y_hat为y的估计值矩阵，
             residual为残差矩阵，p_value为t检验的p值
    """
    # 预处理
    m, n = x.shape  # 样本个数m，变量个数n
    x = np.insert(x, 0, 1, axis=1)  # 加上全1列作为x0（截距项）

    if reg == 'None' or reg == 'Ridge':
        # L2正则化
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

    elif reg == 'LASSO':
        theta = np.random.uniform(-1, 1, [n+1, 1])  # 参数初始化
        ze = np.ones([n+1, 1])  # 记录哪些参数被剔除掉（最终取值为零的参数）
        delta = np.zeros([n+1, 1])  # 梯度初始化
        cost_history = {'epoch': [], 'cost': []}  # 字典记录误差变化

        # 每次迭代依次更新所有维度
        for epoch in range(epochs):

            # 每次只更新一个维度（坐标下降法的精髓之处，克服了“不可导”的问题）
            for k in range(n+1):

                # # 判断该维度的参数是否还需要训练
                # if ze[k, :] == 0:
                #     break

                # 假设函数h(θ)
                h = np.matmul(x, theta)
                # 均方误差损失 + L1正则化
                J = cap.mse(h, y) + lamda*np.linalg.norm(theta[1:n], 1)

                # 坐标下降法（梯度下降法不再适用！！！）
                if k == 0:
                    delta[0, :] = 1/m * np.matmul(x.T[0, :], h-y)  # theta0不加正则化

                else:
                    delta[k, :] = 1/m * np.matmul(x.T[k, :], h-y) + lamda * np.sign(theta[k, :])

                # 参数更新
                theta[k, :] = theta[k, :] - alpha * delta[k, :]

                # ***见调试记录1
                # # 当变化程度很小时，将参数设置为0（即从模型中剔除掉）
                # temp = y - np.dot(x, theta) + np.multiply(x[:, k], theta[k]).reshape(-1, 1)
                # pk = -2 * np.sum(x[:, k] * temp)
                # if lamda >= pk >= -lamda:
                #     ze[k, :] = 0

            # 记录误差cost
            cost_history['epoch'].append(epoch)
            cost_history['cost'].append(J)

        # plt.plot(cost_history['epoch'], cost_history['cost'])
        # plt.show()

        beta_hat = theta  # 参数估计值
        y_hat = np.matmul(x, beta_hat).reshape(-1, 1)  # y的估计值
        residual = y_hat - y  # 残差
        p_value = []

    else:
        print('Error !')

    resi_plot(residual, reg+'回归')  # 残差图
    resi_sum(residual, reg+'回归')  # 残差平方和

    return beta_hat, y_hat, p_value


# 调用sklearn库函数实现LASSO
def LR_sklearn(x, y, alpha=0.01):
    """
    :param x: (m,n)矩阵，决策变量的值
    :param y: (m,1)矩阵，输出的值
    :param alpha: 学习率
    :return:
    """

    LR_model = Lasso(alpha=alpha)  # 创建Lasso模型
    res = LR_model.fit(x, y)  # 模型拟合
    beta_hat = np.insert(res.coef_, 0, res.intercept_).reshape(-1, 1)  # 提取训练完成的参数值
    y_hat = LR_model.predict(x).reshape(-1, 1)  # 预测y值
    residual = y - y_hat  # 残差

    resi_plot(residual, 'LASSO回归(sklearn库)')  # 残差图
    resi_sum(residual, 'LASSO回归(sklearn库)')  # 残差平方和

    return beta_hat, y_hat

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

    beta_hat = list((w1, b1, w2, b2))
    y_hat = network_out  # y的估计值
    residual = output_delta  # 残差

    resi_plot(residual, reg='BP神经网络')  # 残差图
    resi_sum(residual, reg='BP神经网络')  # 残差平方和

    return beta_hat, y_hat


















