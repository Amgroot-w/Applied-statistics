"""
《应用统计与凸优化》9.17作业：
（1）构造函数，求指数分布中λ的区间估计；
（2）生成100个/1000个指数分布随机数，代入函数中计算区间估计。

学号： 2000900
姓名： 张敬川

调试记录：
1. 生成1000个随机数时，计算得到的置信区间包含总体λ值的概率要大于生成100个随机数时，
但是，仍偶尔会出现λ没有位于置信区间中的情况。调试发现，随机数个数越多，计算得到的置
信区间包含λ的可能性越高，在10000个随机数时，基本上每次都在置信区间之中。（当然也可
以升高置信度使λ尽可能的落入置信区间内）
2. 置信度alpha设置的越高，置信区间的精度越低（即上、下置信界的差值越大）；
   置信度alpha设置的越低，置信区间的精度越高（即上、下置信界的差值越小）。

"""
import numpy as np
from scipy import stats


# %% 求λ的区间分布的函数
def interval_estimate(_specimen, _alpha):
    """
    :param _specimen: 样本
    :param _alpha: 置信水平
    :return: 置信区间 (theta_down, theta_up)
    """
    n = _specimen.size  # 样本个数
    mean = _specimen.mean()  # 样本均值
    bias = stats.norm.cdf(_alpha/2) / np.sqrt(n)  # 计算中间项
    theta_down = (1 - bias) / mean  # 下置信界
    theta_up = (1 + bias) / mean  # 上置信界

    return np.array([theta_down, theta_up])  # 返回置信区间


# %% 生成随机数并求区间估计
alpha = 0.05  # 置信水平
lamda = 2  # 总体的λ值
print("\n总体的λ = %d" % lamda)

for num in list([100, 1000, 5000, 10000]):
    # 生成num个服从E(λ)的随机数
    '''
    注意：传入的参数是1/λ而不是λ!!!
    '''
    data = np.random.exponential(1/lamda, num)
    # 求区间估计
    interval = interval_estimate(data, _alpha=alpha)
    # 展示结果
    print("%d个随机数，λ的置信度为 %d%% 的置信区间为：(%.4f, %.4f)"
          % (num, 100*(1-alpha), interval[0], interval[1]))



