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

​		聚类分析是根据在数据中发现的描述对象及其关系的信息，将数据对象分组。目的是，组内的对象相互之间是相似的（相关的），而不同组中的对象是不同的（不相关的）。组内相似性越大，组间差距越大，说明聚类效果越好。

​		说人话就是，根据n维的变量，把n维相似度更高的几个变量定义成一个类别，重点在于对objects之间距离（变量的相似度）的定义，聚类的不同方法也只不过是对距离的不同定义而已。

​		这个东西非常之复杂（代码反正我写不出来。。。），我现在只能大致知道这个东西什么情况下能用，但是使用代码还没搞懂。因为存在非常多种聚类的方法，针对不同的模型建立，需要进行完全不同的聚类分析。但是基本的k-means/Hierarchical，rh哥哥给的网站上都有模版，所以只能到时候慢慢搞了。。。

