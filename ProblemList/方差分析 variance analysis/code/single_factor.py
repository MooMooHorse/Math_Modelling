import numpy as np
from scipy.stats import f as F

np.random.seed(114514)
raw_data = np.random.randint(-10, 20, size=(int(5e3), int(5e5)))

row_n = raw_data.shape[0]
column_n = raw_data.shape[1]
"""
随机生成三组数据，每组8个样本
取消下面注释的一行来模拟存在组间差异
"""
#raw_data[0] = np.random.randint(-5, 20, size=(1, column_n))

print("raw data is:\n", raw_data, '\n')

gp_average = raw_data.mean(axis=1)
print('group average is:', gp_average, '\n')

intraclass_variance = raw_data.var(axis=1, ddof=0).mean()
print('intraclass variance is:', intraclass_variance, '\n')

Variance_between_groups = gp_average.var(ddof=0)
print('Variance between groups is:', Variance_between_groups, '\n')

F_examine = Variance_between_groups /\
    intraclass_variance*((column_n-1)*row_n)/(row_n-1)
print("Factor F is:", F_examine, '\n')


alpha = 0.05  # 调整显著性水平，值越小判定越严格
P = F.pdf(F_examine, (column_n-1)*row_n, (row_n-1))
if (P <= alpha):
    print("F_examine passed!", end='')
else:
    print("F_examine not passed!", end='')
print(" P is:", P)


"""
附上运行数据（分别是注释掉和没注释掉的版本）

raw data is:
 [[  9  17   1 ...  -6  11   7]
 [ -5  16   5 ...   0  14   1]
 [ 15  12 -10 ...   7  19  19]
 ...
 [  6  -3  12 ...   6   2   0]
 [ -6  -3  -8 ...  13   5  16]
 [ -5   8  14 ...  11  -2   3]]

group average is: [4.491784 4.503554 4.502244 ... 4.508866 4.479344 4.506512] 

intraclass variance is: 74.91705831758699 

Variance between groups is: 0.00014849524033162164

Factor F is: 0.9912605460372736

F_examine not passed! P is: 18.268008620027242



raw data is:
 [[ 10   3  18 ...   3  14   2]
 [ -5  16   5 ...   0  14   1]
 [ 15  12 -10 ...   7  19  19]
 ...
 [  6  -3  12 ...   6   2   0]
 [ -6  -3  -8 ...  13   5  16]
 [ -5   8  14 ...  11  -2   3]]

group average is: [7.00207  4.503554 4.502244 ... 4.508866 4.479344 4.506512] 

intraclass variance is: 74.91245280343053 

Variance between groups is: 0.0013999645353746555

Factor F is: 9.34585460361687

F_examine passed! P is: 0.0
"""
