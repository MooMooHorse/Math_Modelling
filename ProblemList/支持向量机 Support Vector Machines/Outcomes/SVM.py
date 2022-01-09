import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
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
plt.show()