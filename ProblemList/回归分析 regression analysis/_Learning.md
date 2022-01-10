# 回归分析



没有任何新的东西



* **贝叶斯公式**

  * https://www.zhihu.com/question/19725590

* **最大似然估计**

  * https://www.matongxue.com/madocs/447/
  * 可以发现，就是通过$k$次实验得到的结果（曲线），来找到参数$\theta$的取值
  * 在 $k$ 等于1的情况下，实际上就是找到一个 $\theta$ 使**已知**目标概率最大。
  * 注意下这个很能帮助理解的小程序
    * ![image-20220110143413377](https://s2.loli.net/2022/01/10/78tOqMSoi5CY96U.png)
    * 这里很容易被弄晕的是$\theta$和推测出的$\theta$之间的关系，
      * $\theta$ 是你用于生成事件（曲线）
      * 推测出的$\theta$是根据曲线拟合找出的最可能的参数
      * 所以他的思路是用$\theta$验证推测出的$\theta$

* **最小二乘法**

  * https://www.zhihu.com/question/37031188
  * 注意到在已知其正确性之前，实际上我们应该是最大化误差概率的乘积
  * 结论就是，如果误差概率是正态分布，那么最大概率的乘积取的点和最小方差取得点是一致的。

* **回归算法**

  >  拟合并不特指某一种方法，指的是对一些数据，按其规律方程化，而其方程化的方法有很多，回归只是其中一种方法，还有[指数平滑](https://www.zhihu.com/search?q=指数平滑&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A"29409845"})这样简单一些的方法，或者ARIMA，VAR，等等各种复杂一点的方法。

  靠，所以回归就是一种拟合数据的方式啊，突然想起来CS101好像弄过。

  * 建议先去看神经网络
    * 因为神经网络就是回归

  因为我们建模如果一类问题能被一个方式统一解决，那么就不用再解决了。

  实现如下(ml实现，不是解析实现)

  https://zhuanlan.zhihu.com/p/53977691

  ```python
  #作者：DsFunStudio
  #链接：https://zhuanlan.zhihu.com/p/53977691
  
  import numpy as np
  from sklearn.linear_model import LinearRegression
  from sklearn.preprocessing import PolynomialFeatures
  import matplotlib.pyplot as plt
  import seaborn as sns
  sns.set()
  
  X_train = [[6], [8], [10], [14], [18]]
  y_train = [[7], [9], [13], [17.5], [18]]
  X_test = [[6], [8], [11], [16]]
  y_test = [[8], [12], [15], [18]]
  
  # 简单线性回归
  model = LinearRegression()
  model.fit(X_train, y_train)
  xx = np.linspace(0, 26, 100)
  yy = model.predict(xx.reshape(xx.shape[0], 1))
  plt.scatter(x=X_train, y=y_train, color='k')
  plt.plot(xx, yy, '-g')
  
  # 多项式回归
  quadratic_featurizer = PolynomialFeatures(degree=2)
  X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
  X_test_quadratic = quadratic_featurizer.fit_transform(X_test)
  model2 = LinearRegression()
  model2.fit(X_train_quadratic, y_train)
  xx2 = quadratic_featurizer.transform(xx[:, np.newaxis])
  yy2 = model2.predict(xx2)
  plt.plot(xx, yy2, '-r')
  
  print('X_train:\n', X_train)
  print('X_train_quadratic:\n', X_train_quadratic)
  print('X_test:\n', X_test)
  print('X_test_quadratic:\n', X_test_quadratic)
  print('简单线性回归R2：', model.score(X_test, y_test))
  print('二次回归R2：', model2.score(X_test_quadratic, y_test));
  ```

  > **步骤详解**
  >
  > 我们来看看在每一步我们都做了什么。
  >
  > 第一步，我们导入了必要的库。
  >
  > 第二步，我们创建了训练集和测试集。
  >
  > 第三步，我们拟合了简单线性回归，并且绘制了预测的直线。
  >
  > 第四步，我们使用`sklearn.preprocessing.PolynomialFeatures`方法，将我们的原始特征集生成了n*3的数据集，其中第一列对应常数项α，相当于x的零次方，因此这一列都是1；第二列对应一次项，因此这一列与我们的原始数据是一致的；第三列对应二次项，因此这一列是我们原始数据的平方。
  >
  > 第四步，我们拿前边用`PolynomialFeatures`处理的数据集做一个[多元线性回归](https://www.zhihu.com/search?q=多元线性回归&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"53977691"})，然后用训练好的模型预测一条曲线，并将其绘制出来。
  >
  > 第五步，输出数据方便理解；输出[模型分数](https://www.zhihu.com/search?q=模型分数&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"53977691"})用于对比效果。
  >
  > 看到这里你可能已经明白了，多项式回归虽然拟合了多项式曲线，但其本质仍然是线性回归，只不过我们将输入的特征做了些调整，增加了它们的多次项数据作为新特征。其实除了多项式回归，我们还可以使用这种方法拟合更多的曲线，我们只需要对原始特征作出不同的处理即可。