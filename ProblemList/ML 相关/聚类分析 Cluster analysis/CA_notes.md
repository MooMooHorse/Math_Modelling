# 总结

## 1. 1主成分分析&因子分析

​		这两个玩意儿其实还是比较好理解的，最主要的区别是——**主成分分析的对象不一定是完全独立的**（更多情况是，相互之间是有一些联系的），**因子分析的对象是完全独立的**（比如人的性别/年龄/姓名），他们干的事情也不一样，主成分分析是在做一件降维的事情，以减小计算开销（可以理解成把几个向量合成一个向量）

​		就美赛而言，因子分析的作用很小，因为题目不会提供给你绝对独立的模型，大多数都是我们自己建立的vague的模型，所以唯一的用了因子分析的模型也算是自己硬凑凑出来的，所以我觉得PCA的使用价值更大

## 1.2 PCA的基本流程

1. 去平均值（假设一共有p的奇怪的向量，把他们的mean全部先调成0）

2. 计算协方差矩阵

3. 用特征值解法求 协方差矩阵 的特征值和特征矩阵

4. 对特征值从大到小排序，选择其中最大的k个。这k个向量应当覆盖所有向量（p个）

5. 至此，把p个向量降维到了k个向量，然后进行一个很抽象的操作（比如说k1 = 0.6力量 + 0.4速度，k2 = 0.5交际能力 + 0.3情商 + 0.2 智商，**就人为**把它定义成k1是武力值，k2是社交能力，就很离谱）

   

6. ```python3
   import numpy as np
   def pca(X,k):#k is the components you want
     #mean of each feature
     n_samples, n_features = X.shape
     mean=np.array([np.mean(X[:,i]) for i in range(n_features)])
     
     #1
     norm_X=X-mean
     
     #2
     scatter_matrix=np.dot(np.transpose(norm_X),norm_X)
     
     #3
     eig_val, eig_vec = np.linalg.eig(scatter_matrix)
     eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
     
     #4
     eig_pairs.sort(reverse=True)
     
     #5
     feature=np.array([ele[1] for ele in eig_pairs[:k]])
     
     #get new data
     data=np.dot(norm_X,np.transpose(feature))
     return data
   ```



## 2.1聚类分析



![0a23e777ba865a853b1a49cc8188e67](https://s2.loli.net/2022/01/12/X6ZW4EISYhG5dAw.png)

> 聚类原本是统计学上的概念，现在属于机器学习中**非监督学习**的范畴，大多都被应用在数据挖掘、数据分析的领域，简单说可以用一个词概括——物以类聚。
>
> 从定义上讲，聚类就是针对大量数据或者样品，根据数据本身的特性研究分类方法，并遵循这个分类方法对数据进行合理的分类，最终将相似数据分为一组，也就是“**同类相同、异类相异**”。
>
> 
>
> 作者：李启方
> 链接：https://zhuanlan.zhihu.com/p/113894809



### 聚类分析解决问题的类型

* 非监督学习

  * 没有明确告诉你分类的标准，也就是训练样本没有答案，需要你自己找答案
  * 与分类(`classification`)的区别是，分类，对于训练样本已经给了类别，从训练的角度来说，你训练的神经网络已经可以得到一个标准了，这种标准可能是`decision tree`,而聚类是不给标准的，而是直接对样本进行**最优化分类**。

* 最后得到的是分类的结果，而不是**quantity**

  

### 实质

​		聚类分析是根据在数据中发现的描述对象及其关系的信息，将数据对象分组。目的是，组内的对象相互之间是相似的（相关的），而不同组中的对象是不同的（不相关的）。组内相似性越大，组间差距越大，说明聚类效果越好。

* 根据n维的变量，把n维相似度更高的几个变量定义成一个类别，**重点在于对objects之间距离（变量的相似度）的定义，聚类的不同方法也只不过是对距离的不同定义而已**。

​		~~这个东西非常之复杂（代码反正我写不出来。。。），我现在只能大致知道这个东西什么情况下能用，但是使用代码还没搞懂。因为存在非常多种聚类的方法，针对不同的模型建立，需要进行完全不同的聚类分析。但是基本的k-means/Hierarchical，rh哥哥给的网站上都有模版，所以只能到时候慢慢搞了。。。~~

### 不同方法的理解

#### 只需要了解分类标准

* 等我们用到的时候，知道具体用哪种后，再看对应材料，快速改动公式引用即可

* 只要知道每种方法解决什么问题就行了（看图&表格），剩下的就是直接调用库。~~最好理解一下，因为写论文，你总得写点理由吧，这个真的挺好理解的。~~



|                         Method name                          |                          Parameters                          |                         Scalability                          |                           Usecase                            |            Geometry (metric used)            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :------------------------------------------: |
| [K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means) |                      number of clusters                      | Very large `n_samples`, medium `n_clusters` with [MiniBatch code](https://scikit-learn.org/stable/modules/clustering.html#mini-batch-kmeans) | General-purpose, even cluster size, flat geometry, not too many clusters, inductive |           Distances between points           |
| [Affinity propagation](https://scikit-learn.org/stable/modules/clustering.html#affinity-propagation) |                  damping, sample preference                  |                 Not scalable with n_samples                  | Many clusters, uneven cluster size, non-flat geometry, inductive | Graph distance (e.g. nearest-neighbor graph) |
| [Mean-shift](https://scikit-learn.org/stable/modules/clustering.html#mean-shift) |                          bandwidth                           |                Not scalable with `n_samples`                 | Many clusters, uneven cluster size, non-flat geometry, inductive |           Distances between points           |
| [Spectral clustering](https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering) |                      number of clusters                      |            Medium `n_samples`, small `n_clusters`            | Few clusters, even cluster size, non-flat geometry, transductive | Graph distance (e.g. nearest-neighbor graph) |
| [Ward hierarchical clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) |           number of clusters or distance threshold           |              Large `n_samples` and `n_clusters`              | Many clusters, possibly connectivity constraints, transductive |           Distances between points           |
| [Agglomerative clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) | number of clusters or distance threshold, linkage type, distance |              Large `n_samples` and `n_clusters`              | Many clusters, possibly connectivity constraints, non Euclidean distances, transductive |            Any pairwise distance             |
| [DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan) |                      neighborhood size                       |         Very large `n_samples`, medium `n_clusters`          | Non-flat geometry, uneven cluster sizes, outlier removal, transductive |       Distances between nearest points       |
| [OPTICS](https://scikit-learn.org/stable/modules/clustering.html#optics) |                  minimum cluster membership                  |          Very large `n_samples`, large `n_clusters`          | Non-flat geometry, uneven cluster sizes, variable cluster density, outlier removal, transductive |           Distances between points           |
| [Gaussian mixtures](https://scikit-learn.org/stable/modules/mixture.html#mixture) |                             many                             |                         Not scalable                         |    Flat geometry, good for density estimation, inductive     |      Mahalanobis distances to  centers       |
| [BIRCH](https://scikit-learn.org/stable/modules/clustering.html#birch) |   branching factor, threshold, optional global clusterer.    |              Large `n_clusters` and `n_samples`              |  Large dataset, outlier removal, data reduction, inductive   |      Euclidean distance between points       |

---

作者：李启方

链接：https://zhuanlan.zhihu.com/p/113894809

---

### 简单了解 K-means

简答拿一个例子介绍一下K-Means聚类法的原理和过程：

**1、确定分组数**

`K-Mcans`聚类法中的K就是分组数，也就是我们希望通过聚类后得到多少个组类。比如我有下面六个数据，想要将这些数据分成两类，那么K=2 。

![img](https://pic3.zhimg.com/v2-40f8a56c23770b109a92186c45b9138a_b.jpg)

**2、随机选择K个值作为数据中心**

这个数据中心的选择是完全随机的，也就是说怎么选择都无所谓，因为这里K=2，所以我们就以A和B两个为数据中心。

为了方便理解，我们可以制作一个[散点](https://www.zhihu.com/search?q=散点&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"113894809"})图，将A、B作为数据中心。

![img](https://pic3.zhimg.com/v2-e92a96b628bf7f888c3996fc5238b3a2_b.jpg)

**3、计算其他数值与数据中心的“距离”**

既然选择了[数据中心](https://www.zhihu.com/search?q=数据中心&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"113894809"})，那么它们的周围一定会有很多相似数据，怎么判断这些数据与其是不是相似呢？

这里我们要引入[欧氏距离](https://www.zhihu.com/search?q=欧氏距离&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"113894809"})的概念，通俗点说欧氏距离就是[多维空间](https://www.zhihu.com/search?q=多维空间&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"113894809"})中各个点之间的绝对距离，表明两点之间的距离远近，其公式为：

![img](https://pic3.zhimg.com/v2-4b4e192cac6a1a0eab17464a10d0c542_b.jpg)

如果是普通的二维数据，这个公式就直接变成了[勾股定理](https://www.zhihu.com/search?q=勾股定理&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"113894809"})，因此我们算出其他6个点距离A和B的距离，谁离得更近，谁与数据中心就是同一类。

![img](https://pic1.zhimg.com/v2-8b9b17951b628f52d3123b0f22bd26d4_b.jpg)

所以，我们可以看出，C-H距离B的距离都比距离A更近，所以第一次分组为：

- 第一组：A
- 第二组：B、C、D、E、F、G、H

**4、重新选择新的数据中心**

得到了第一次分组的结果，我们再重复前两个步骤，重新选择每一组数据的数据中心。

- 第一组只有A，所以A仍然是数据中心；
- 第二组有7个数值，将这个7个数值的平均值作为新的数据中心，我们将其命名为P，计算平均坐标为（5.14 ，5.14）

**5、再次计算其他数据与新数据中心的距离**

还是直接计算勾股定理，计算出其他数据与A和P的欧氏距离，如下：

![img](https://pic1.zhimg.com/v2-d6a0e0003aaf3f78ebd54fa0bdb01194_b.jpg)

我们可以看出这里面有的距离A近，有的距离P近，于是第二次分组为：

- 第一组：A、B
- 第二组：C、D、E、F、G、H

**6、再次重新选择数据中心**

这里就是老规矩了，继续重复前面的操作，将每一组数据的平均值作为数据中心：

- 第一组有两个值，平均坐标为（0.5 ，1），这是第一个新的数据中心，命名为O
- 第二组有六个值，平均值为（5.8 ， 5.6），这是第二个新的数据中心，命名为Q

**7、再次计算其他数据与新数据中心的距离**

![img](https://pic1.zhimg.com/v2-52bd92c39e960899a1196919ff84975c_b.jpg)

这时候我们发现，只有A与B距离O的距离更近，其他6个数据都距离Q更近，因此第三次分组为：

- 第一组：A、B
- 第二组：C、D、E、F、G、H

经过这次计算我们发现分组情况并没有变化，这就说明我们的计算收敛已经结束了，不需要继续进行分组了，最终数据成功按照[相似性](https://www.zhihu.com/search?q=相似性&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"113894809"})分成了两组。

**8、方法总结**

简单来说，我们一次次重复这样的**选择数据中心-计算距离-分组-再次选择数据中心**的流程，直到我们分组之后所有的数据都不会再变化了，也就得到了最终的聚合结果。

#### 完整的教程

https://scikit-learn.org/stable/modules/clustering.html

### Clustering performance evaluation

* 这个挺重要，写在里面肯定加分项
