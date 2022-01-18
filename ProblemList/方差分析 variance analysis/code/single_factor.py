import numpy as np

np.random.seed(123)
raw_data = np.random.randint(-10, 50, size=(10, 8))
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
