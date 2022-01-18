import numpy as np
from scipy.stats import f as F

np.random.seed(123)
raw_data = np.random.randint(-10, 50, size=(3, 8))
'''
随机生成三组数据，每组8个样本
取消下面注释的一行来模拟存在组间差异
'''
#raw_data[0] = np.linspace(1, 10, 8)


print("raw data is:\n", raw_data, '\n')

column_n = raw_data.shape[1]
row_n = raw_data.shape[0]

gp_average = raw_data.mean(axis=1)
print('group average is:', gp_average, '\n')

intraclass_variance = raw_data.var(axis=1, ddof=1).mean()
print('intraclass variance is:', intraclass_variance, '\n')

Variance_between_groups = gp_average.var(ddof=1)
print('Variance between groups is:', Variance_between_groups, '\n')

F_examine = Variance_between_groups / \
    intraclass_variance*((column_n-1)*row_n)/(row_n-1)
print("Factor F is:", F_examine, '\n')


alpha = 0.05  # 调整显著性水平，值越小判定越严格
P = F.pdf(F_examine, (column_n-1)*row_n, (row_n-1))
if (P <= alpha):
    print("F_examine passed!", end='')
else:
    print("F_examine not passed!", end='')
print(" P is:", P)
