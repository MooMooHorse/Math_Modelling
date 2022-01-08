# Support Vector Machines Notes

## 需要用到的知识

### 高数中的拉格朗日乘子（已学）

### 矩阵分析里的拉格朗日乘子

* #### 包含多个等式约束的最优化问题

函数$f(x)=x^T\omega+b$存在约束条件$g_k(x)=x^T\omega_k+b_k=0,\space k=1,2\cdots ,N$

则求最小值问题可构建拉格朗日函数

$$L\left(x,\lambda\right)=f(x)+\sum_{k=1}^{N}\lambda_kg_k(x)$$

求偏导后联立方程

$$\frac{\partial f(x)}{x*}+\sum_{k=1}^N\frac{\partial}{\partial x*}\lambda_kg_k(x)=0$$

* #### 含有不等式约束（拉格朗日对偶问题）
