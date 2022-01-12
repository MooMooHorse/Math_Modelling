# Support Vector Machines Notes

https://blog.csdn.net/weixin_44378835/article/details/110732412

https://blog.csdn.net/liweibin1994/article/details/77504210

## 定义

对于线性可分的两类问题，其分类决策边界为一 $n$ 维特征空间中的超平面 $H$，一般情况下会有无穷多个解。当我们确定了一个解对应的权向量 $\omega$，超平面的斜率和朝向就是确定的了，可以在一定的范围内平移超平面 $H$，只要不达到或者越过两类中距离 $H$ 最近的样本，分类决策边界都可以正确地实现线性分类。所以，任何一个求解得到的权向量 $\omega$，都会带来一系列平行的分类决策边界，其可平移的范围具有一定的宽度，称为分类间隔 $d$ (Margin of Classification)。

![img](https://img-blog.csdnimg.cn/20201206122540665.png)

显然，当我们改变 $\omega$，使分类决策边界的斜率和朝向随之变化时，我们得到的分类间隔是不同的。那么，分类间隔究竟是大还是小更好呢？当然是越大越好。因为分类间隔越大，两类样本做分类决策时的余量也就越大，由于样本采集所带来的特征值误差所造成的分类错误也就越少。所以，在所有能够实现正确分类的权向量 $\omega$ 中，去求取到分类间隔最大的一个 $\omega^*$ ，就是对线性分类器进行优化求解的一个很好的指标。这就是“支持向量机”这种线性分类器训练算法的出发点。

## 需要用到的知识

###    高数中的拉格朗日乘子（已学）

******************

<font size=5 color=red>**这段实在是太复杂了，建议直接跳过直接看结论!!!**</font>

### 矩阵分析里的拉格朗日乘子

* ####    包含多个等式约束的最优化问题

函数$f(x)=x^T\omega+b$存在约束条件$g_k(x)=x^T\omega_k+b_k=0,\space k=1,2\cdots ,N$

则求最小值问题可构建拉格朗日函数

$$L\left(x,\lambda\right)=f(x)+\sum_{k=1}^{N}\lambda_kg_k(x)$$

求偏导后联立方程

$$\frac{\partial f(x)}{\partial x*}+\sum_{k=1}^N\frac{\partial}{\partial x*}\lambda_kg_k(x)=0$$

* #### 含有不等式约束（拉格朗日对偶问题）

两条性质：

1. $f(x)=\max\{x_1,x_2\cdots,x_n\}$在$\R^n$上是凸函数。
2. $f(x)=\min\{x_1,x_2\cdots,x_n\}$在$\R^n$上是凹函数。

证明略

**$$\vdots$$**

总之就是用一堆定理推导出了一个通式！不要在上面浪费时间！

******************

## 结论

### 一般的支持向量机

$f(x)=\omega^Tx+b$在$g(x)<0$时满足$f(x)\le-1$，在$g(x)>0$时满足$f(x)\ge1$，求$\omega$与$b$

正如定义中所说，这个问题的解有无数个，而下式可以给出一种最优解：

$$\max_{\alpha}{\left(\sum_{i=1}^n\alpha^{(i)}-\frac12\sum_{i=1}^{n}\left(\sum_{j=1}^nsgn(g(x^{(i)}))sgn(g(x^{(j)}))x^{(i)T}x^{(j)}\right)\right)}$$

$$s.t.\space\forall i.\alpha^{(i)}\ge0$$

​		$$\sum_{i=1}^{n}a^{(i)}sgn\left(g(x^{(i)})\right)\left(f(x^{(i)})-1\right)=0$$

解出$\alpha$后~~(还不知道怎么解)~~，带入下式解$\omega$

$$\omega=\sum_{i=1}^{n_s}sgn(g(x_s^{(i)})\alpha^{(i)}x_s^{
(i)})$$

$x_s$是所有的支持向量（边界上的向量）

解出$\omega$后随便代几个$x$进去，把$b$解出来就算做完了。

### 非线性分类(核函数)以及松弛变量

核函数的思想是将维度升高来解决非线性的问题（因为上式$\max_\alpha$中实际只用到了点积$x^{(i)T}x^{(j)}$，所以就直接构建一个核函数来算点积的值了）

![20170824165350942](https://s2.loli.net/2022/01/08/8g5PKROrdqWyIVN.png)

松弛变量则是用来解决存在部分噪点但本质还是线性可分的问题

![20201205223525843](https://s2.loli.net/2022/01/08/KlYHwkcydCx6sU7.png)

## 实现

我认为只要大致了解SVM的运作原理即可，至于实现，python上有现成的包可用（好耶！我就是调包侠）

确保安装对应的包：`pip install sklearn`

### 百度百科上的示例程序

```python
# 导入模块
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
% matplotlib inline
# 鸢尾花数据
iris = datasets.load_iris()
X = iris.data[:, :2] # 为便于绘图仅选择2个特征
y = iris.target
# 测试样本（绘制分类区域）
xlist1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
xlist2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 200)
XGrid1, XGrid2 = np.meshgrid(xlist1, xlist2)
# 非线性SVM：RBF核，超参数为0.5，正则化系数为1，SMO迭代精度1e-5, 内存占用1000MB
svc = svm.SVC(kernel='rbf', C=1, gamma=0.5, tol=1e-5, cache_size=1000).fit(X, y)
# 预测并绘制结果
Z = svc.predict(np.vstack([XGrid1.ravel(), XGrid2.ravel()]).T)
Z = Z.reshape(XGrid1.shape)
plt.contourf(XGrid1, XGrid2, Z, cmap=plt.cm.hsv)
plt.contour(XGrid1, XGrid2, Z, colors=('k',))
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', linewidth=1.5, cmap=plt.cm.hsv)
```

### 相关文档：

https://blog.csdn.net/u012526003/article/details/79088214

### 注解：

RBF即高斯核，相对的线性是poly

调参（炼丹）

## 论文相关

论文中SVM只是一个用来预测index value of years的辅助工具`（见5.2.1节）`，感觉是凑数用的。
