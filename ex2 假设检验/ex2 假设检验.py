"""
《应用统计与凸优化》9.24作业：
（1）构造函数，做 wald test；
（2）生成50个、100个二项分布随机数，代入函数中计算p值。

学号： 2000900
姓名： 张敬川

调试记录：

"""
import numpy as np
from scipy import stats


# 求二项分布参数p的最大似然估计值
def binomial_MLE(sample, N):

    """
    :param sample: 二项分布的样本
    :param N: 二项分布的参数N（实验次数）
    :return: 参数p的最大似然估计值
    """
    return np.mean(sample) / N


# 求二项分布的Fisher信息量
def binomial_fisher(sample, N):
    """
    :param sample: 二项分布的样本
    :param N: 二项分布的参数N（实验次数）
    :return: Fisher信息量
    """
    # 求p的最大似然估计
    P = binomial_MLE(sample, N)

    return N / (P * (1-P))


# 二项分布参数p的wald检验
def wald_test(sample, N, theta0):
    """
    :param sample: 二项分布的样本
    :param N: 二项分布的参数N（实验次数）
    :param theta0: 假设检验的p值（已知）
    :return: p-value
    """
    n = sample.size  # 样本个数
    theta_MLE = binomial_MLE(sample, N)         # 求p的最大似然估计
    I_theta = binomial_fisher(sample, N)        # 求二项分布的Fisher信息量
    Tn = n * I_theta * (theta_MLE - theta0)**2  # Tn表征了theta_MLE和theta0的接近程度
    C = Tn                                      # C为H0拒绝域的未知参数
    q_alpha = C                                 # q_alpha为卡方分布的（1-alpha）分位点
    chi2_p = stats.chi2.cdf(q_alpha, 1)         # 已知分位点为q_alpha，自由度为1，求卡方分布的p值
    p_value = 1 - chi2_p                        # 转换为右侧尾部的概率

    if p_value < 0.05:
        print("%d个随机数，p-value为%.8f；拒绝原假设H0，即：p ≠ 0.5" % (n, p_value))
    else:
        print("%d个随机数，p-value为%.8f；接受原假设H0，即：p = 0.5" % (n, p_value))

    return p_value


if __name__ == "__main__":

    sample1 = np.random.binomial(20, 0.45, 50)   # 50个随机数
    pval1 = wald_test(sample1, N=20, theta0=0.5)

    sample2 = np.random.binomial(20, 0.45, 100)  # 100个随机数
    pval2 = wald_test(sample2, N=20, theta0=0.5)











