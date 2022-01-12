## 1. 数据降维

降维就是一种对高维度特征数据预处理方法。降维是将高维度的数据保留下最重要的一些特征，去除噪声和不重要的特征，从而实现提升数据处理速度的目的。在实际的生产和应用中，降维在一定的信息损失范围内，可以为我们节省大量的时间和成本。降维也成为应用非常广泛的数据预处理方法。

降维具有如下一些优点：

- 使得数据集更易使用。
- 降低算法的计算开销。
- 去除噪声。
- 使得结果容易理解。

降维的算法有很多，比如[奇异值分解(SVD)](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/Dv51K8JETakIKe5dPBAPVg)、主成分分析(PCA)、因子分析(FA)、独立成分分析(ICA)。

## 3. PCA原理详解（xb概括）

简单来说就是把n维的变量，降维到k维（n>k）

可以想象是有一个m*n的矩阵，n是决定因素，m的各个决定因素下的值的向量（可以想象成**n个线性不相关的向量**），PCA做的事情就是把这n个线性不相关的中的可以被其他向量（在区间内大致）合成的  **弱因子向量**  删掉，以达到上面👆数据降维。

对于各个向量的印象分析，PCA引入了**特征值和特征向量**的概念，本篇文章里面，只介绍了怎么把n维变成n-1的方法，还是比较容易理解的，n->k有待研究，我看看往届论文

特征值便于理解可以想象成 对结果的影响比例，特征值越大，影响力越大





## 3.1 PCA的概念

PCA(Principal Component Analysis)，即主成分分析方法，是一种使用最广泛的数据降维算法。PCA的主要思想是将n维特征映射到k维上，这k维是全新的正交特征也被称为主成分，是在原有n维特征的基础上重新构造出来的k维特征。PCA的工作就是从原始的空间中顺序地找一组相互正交的坐标轴，新的坐标轴的选择与数据本身是密切相关的。其中，第一个新坐标轴选择是原始数据中方差最大的方向，第二个新坐标轴选取是与第一个坐标轴正交的平面中使得方差最大的，第三个轴是与第1,2个轴正交的平面中方差最大的。依次类推，可以得到n个这样的坐标轴。通过这种方式获得的新的坐标轴，我们发现，大部分方差都包含在前面k个坐标轴中，后面的坐标轴所含的方差几乎为0。于是，我们可以忽略余下的坐标轴，只保留前面k个含有绝大部分方差的坐标轴。事实上，这相当于只保留包含绝大部分方差的维度特征，而忽略包含方差几乎为0的特征维度，实现对数据特征的降维处理。

**思考：**我们如何得到这些包含最大差异性的主成分方向呢？

**答案：**事实上，通过计算数据矩阵的协方差矩阵，然后得到协方差矩阵的特征值特征向量，选择特征值最大(即方差最大)的k个特征所对应的特征向量组成的矩阵。这样就可以将数据矩阵转换到新的空间当中，实现数据特征的降维。

**以上即是我在原理详解里的分析**



由于得到协方差矩阵的特征值特征向量有两种方法：特征值分解协方差矩阵、奇异值分解协方差矩阵，所以PCA算法有两种实现方法：

基于特征值分解协方差矩阵实现PCA算法、基于SVD分解协方差矩阵实现PCA算法。



既然提到协方差矩阵，那么就简单介绍一下方差和协方差的关系。然后概括介绍一下特征值分解矩阵原理、奇异值分解矩阵的原理。概括介绍是因为在我之前的[《机器学习中SVD总结》](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/Dv51K8JETakIKe5dPBAPVg)文章中已经详细介绍了特征值分解原理和奇异值分解原理，这里就不再重复讲解了。可以看我的

[《机器学习中SVD总结》](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/Dv51K8JETakIKe5dPBAPVg)文章。地址：[机器学习中SVD总结](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/Dv51K8JETakIKe5dPBAPVg)

**这篇文章主要讲的是，求特征值和特征向量用的方法，有点诡异，有待研究，xb看完实例之后，再来补充**

## 3.2 协方差和散度矩阵

样本均值：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbar%7Bx%7D%3D%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%7Bx_%7Bi%7D%7D)

样本方差：

![[公式]](https://www.zhihu.com/equation?tex=S%5E%7B2%7D%3D%5Cfrac%7B1%7D%7Bn-1%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Cleft%28+x_%7Bi%7D-%5Cbar%7Bx%7D+%5Cright%29%5E2%7D)

样本X和样本Y的协方差：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%2A%7D+Cov%5Cleft%28+X%2CY+%5Cright%29%26%3DE%5Cleft%5B+%5Cleft%28+X-E%5Cleft%28+X+%5Cright%29+%5Cright%29%5Cleft%28+Y-E%5Cleft%28+Y+%5Cright%29+%5Cright%29+%5Cright%5D+%5C%5C+%26%3D%5Cfrac%7B1%7D%7Bn-1%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%28x_%7Bi%7D-%5Cbar%7Bx%7D%29%28y_%7Bi%7D-%5Cbar%7By%7D%29%7D+%5Cend%7Balign%2A%7D)

由上面的公式，我们可以得到以下结论：

(1) 方差的计算公式是针对一维特征，即针对同一特征不同样本的取值来进行计算得到；而协方差则必须要求至少满足二维特征；方差是协方差的特殊情况。

(2) 方差和协方差的除数是n-1,这是为了得到方差和协方差的无偏估计。

协方差为正时，说明X和Y是正相关关系；协方差为负时，说明X和Y是负相关关系；协方差为0时，说明X和Y是相互独立。Cov(X,X)就是X的方差。当样本是n维数据时，它们的协方差实际上是协方差矩阵(对称方阵)。例如，对于3维数据(x,y,z)，计算它的协方差就是：

![[公式]](https://www.zhihu.com/equation?tex=Cov%28X%2CY%2CZ%29%3D%5Cleft%5B+%5Cbegin%7Bmatrix%7D+Cov%28x%2Cx%29+%26+Cov%28x%2Cy%29%26Cov%28x%2Cz%29+%5C%5C+Cov%28y%2Cx%29%26Cov%28y%2Cy%29%26Cov%28y%2Cz%29+%5C%5C+Cov%28z%2Cx%29%26Cov%28z%2Cy%29%26Cov%28z%2Cz%29+%5Cend%7Bmatrix%7D+%5Cright%5D)

散度矩阵定义为：

![img](https://pic2.zhimg.com/80/v2-49ec8132e7c3459b8269475c912f1ced_1440w.jpg)散度矩阵

对于数据X的散度矩阵为 ![[公式]](https://www.zhihu.com/equation?tex=XX%5ET) 。其实协方差矩阵和散度矩阵关系密切，散度矩阵就是协方差矩阵乘以（总数据量-1)。因此它们的**特征值**和**特征向量**是一样的。这里值得注意的是，散度矩阵是**SVD奇异值分解**的一步，因此PCA和SVD是有很大联系。





## 3.3 特征值分解矩阵原理

(1) 特征值与特征向量

如果一个向量v是矩阵A的特征向量，将一定可以表示成下面的形式：

![[公式]](https://www.zhihu.com/equation?tex=Av%3D%5Clambda+v)

其中，λ是特征向量v对应的特征值，一个矩阵的一组特征向量是一组正交向量。

(2) 特征值分解矩阵

对于矩阵A，有一组特征向量v，将这组向量进行正交化单位化，就能得到一组正交单位向量。**特征值分解**，就是将矩阵A分解为如下式：

![[公式]](https://www.zhihu.com/equation?tex=A%3DQ%5CSigma+Q%5E%7B-1%7D)

其中，Q是矩阵A的特征向量组成的矩阵，![[公式]](https://www.zhihu.com/equation?tex=%5CSigma)则是一个对角阵，对角线上的元素就是特征值。

具体了解这一部分内容看我的[《机器学习中SVD总结》](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/Dv51K8JETakIKe5dPBAPVg)文章。地址：[机器学习中SVD总结](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/Dv51K8JETakIKe5dPBAPVg)









## 3.4 SVD分解矩阵原理

奇异值分解是一个能适用于任意矩阵的一种分解的方法，对于任意矩阵A总是存在一个奇异值分解：

![[公式]](https://www.zhihu.com/equation?tex=A%3DU%5CSigma+V%5E%7BT%7D)

假设A是一个m*n的矩阵，那么得到的U是一个m*m的方阵，U里面的正交向量被称为左奇异向量。Σ是一个m*n的矩阵，Σ除了对角线其它元素都为0，对角线上的元素称为奇异值。 ![[公式]](https://www.zhihu.com/equation?tex=V%5E%7BT%7D) 是v的转置矩阵，是一个n*n的矩阵，它里面的正交向量被称为右奇异值向量。而且一般来讲，我们会将Σ上的值按从大到小的顺序排列。

**SVD分解矩阵A的步骤：**

(1) 求![[公式]](https://www.zhihu.com/equation?tex=AA%5ET) 的特征值和特征向量，用单位化的特征向量构成 U。

(2) 求 ![[公式]](https://www.zhihu.com/equation?tex=A%5ETA) 的特征值和特征向量，用单位化的特征向量构成 V。

(3) 将 ![[公式]](https://www.zhihu.com/equation?tex=AA%5ET) 或者 ![[公式]](https://www.zhihu.com/equation?tex=A%5ETA) 的特征值求平方根，然后构成 Σ。

具体了解这一部分内容看我的[《机器学习中SVD总结》](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/Dv51K8JETakIKe5dPBAPVg)文章。地址：[机器学习中SVD总结](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/Dv51K8JETakIKe5dPBAPVg)









## 3.5 PCA算法两种实现方法

## **(1) 基于特征值分解协方差矩阵实现PCA算法**

输入：数据集 ![[公式]](https://www.zhihu.com/equation?tex=X%3D%5Cleft%5C%7B+x_%7B1%7D%2Cx_%7B2%7D%2Cx_%7B3%7D%2C...%2Cx_%7Bn%7D+%5Cright%5C%7D) ，需要降到k维。

\1) 去平均值(即去中心化)，即每一位特征减去各自的平均值。

\2) 计算协方差矩阵 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7Bn%7DXX%5ET),注：这里除或不除样本数量n或n-1,其实对求出的特征向量没有影响。

\3) 用特征值分解方法求协方差矩阵![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7Bn%7DXX%5ET) 的特征值与特征向量。

\4) 对特征值从大到小排序，选择其中最大的k个。然后将其对应的k个特征向量分别作为行向量组成特征向量矩阵P。

\5) 将数据转换到k个特征向量构建的新空间中，即Y=PX。

**总结：**

1)关于这一部分为什么用 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7Bn%7DXX%5ET) ,这里面含有很复杂的线性代数理论推导，想了解具体细节的可以看下面这篇文章。

[CodingLabs - PCA的数学原理](https://link.zhihu.com/?target=http%3A//blog.codinglabs.org/articles/pca-tutorial.html)

2)关于为什么用特征值分解矩阵，是因为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B1%7D%7Bn%7DXX%5ET) 是方阵，能很轻松的求出特征值与特征向量。当然，用奇异值分解也可以，是求特征值与特征向量的另一种方法。

**举个例子：**

![[公式]](https://www.zhihu.com/equation?tex=X%3D%5Cleft%28+%5Cbegin%7Bmatrix%7D+-1+%26+-1+%260%262%260%5C%5C+-2%260%260%261%261+%5Cend%7Bmatrix%7D+%5Cright%29)

以X为例，我们用PCA方法将这两行数据降到一行。

1)因为X矩阵的每行已经是零均值，所以不需要去平均值。

2)求协方差矩阵：

![[公式]](https://www.zhihu.com/equation?tex=C%3D%5Cfrac%7B1%7D%7B5%7D%5Cleft%28+%5Cbegin%7Bmatrix%7D+-1%26-1%260%262%260%5C%5C+-2%260%260%261%261+%5Cend%7Bmatrix%7D+%5Cright%29+%5Cleft%28+%5Cbegin%7Bmatrix%7D+-1%26-2%5C%5C+-1%260%5C%5C+0%260%5C%5C+2%261%5C%5C+0%261+%5Cend%7Bmatrix%7D+%5Cright%29+%3D+%5Cleft%28+%5Cbegin%7Bmatrix%7D+%5Cfrac%7B6%7D%7B5%7D%26%5Cfrac%7B4%7D%7B5%7D%5C%5C+%5Cfrac%7B4%7D%7B5%7D%26%5Cfrac%7B6%7D%7B5%7D+%5Cend%7Bmatrix%7D+%5Cright%29)

3)求协方差矩阵的特征值与特征向量。

求解后的特征值为：

![[公式]](https://www.zhihu.com/equation?tex=%5Clambda_%7B1%7D%3D2%EF%BC%8C%5Clambda_%7B2%7D%3D%5Cfrac%7B2%7D%7B5%7D)

对应的特征向量为：

![[公式]](https://www.zhihu.com/equation?tex=c_%7B1%7D+%5Cleft%28+%5Cbegin%7Bmatrix%7D+1%5C%5C+1+%5Cend%7Bmatrix%7D+%5Cright%29) , ![[公式]](https://www.zhihu.com/equation?tex=c_%7B2%7D+%5Cleft%28+%5Cbegin%7Bmatrix%7D+-1%5C%5C+1+%5Cend%7Bmatrix%7D+%5Cright%29)

其中对应的特征向量分别是一个通解， ![[公式]](https://www.zhihu.com/equation?tex=c_%7B1%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=c_%7B2%7D) 可以取任意实数。那么标准化后的特征向量为：

![[公式]](https://www.zhihu.com/equation?tex=+%5Cleft%28+%5Cbegin%7Bmatrix%7D+%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D%5C%5C+%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D+%5Cend%7Bmatrix%7D+%5Cright%29) , ![[公式]](https://www.zhihu.com/equation?tex=+%5Cleft%28+%5Cbegin%7Bmatrix%7D+-%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D%5C%5C+%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D+%5Cend%7Bmatrix%7D+%5Cright%29)

4)矩阵P为：

![[公式]](https://www.zhihu.com/equation?tex=P%3D%5Cleft%28+%5Cbegin%7Bmatrix%7D+%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D%26%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D%5C%5C+-%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D%26%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D%5C%5C+%5Cend%7Bmatrix%7D+%5Cright%29)

5)最后我们用P的第一行乘以数据矩阵X，就得到了降维后的表示：

![[公式]](https://www.zhihu.com/equation?tex=Y%3D%5Cleft%28+%5Cbegin%7Bmatrix%7D+%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D+%26+%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D+%5Cend%7Bmatrix%7D+%5Cright%29+%5Cleft%28+%5Cbegin%7Bmatrix%7D+-1+%26+-1%26+0%262%260%5C%5C+-2%260%260%261%261+%5Cend%7Bmatrix%7D+%5Cright%29+%3D+%5Cleft%28+%5Cbegin%7Bmatrix%7D+-%5Cfrac%7B3%7D%7B%5Csqrt%7B2%7D%7D+%26+-+%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D+%260%26%5Cfrac%7B3%7D%7B%5Csqrt%7B2%7D%7D+%26+-%5Cfrac%7B1%7D%7B%5Csqrt%7B2%7D%7D+%5Cend%7Bmatrix%7D+%5Cright%29)







![img](https://pic2.zhimg.com/80/v2-f5b0a7ae6d0b400e65220a02a0f0c1c1_1440w.jpg)

至此，可以把一个二维的矩阵降维到一维。













# application

2014MCM 评定最佳教练

29696.pdf



1. 首先自己定义了对于评价教练的12个因子Xi，以及3个主成分Zi
2. 然后，找了一个小样本60个来自NCAA的教练，然后自己给他们打了个分（这个就离谱）
3. 列出m个（m<12）sigama:amj*Xj=Zm 的线性方程组（具体看论文）
4. 进行PCA，求各个因子的特征值
5. 然后把拟合条件最优的三个Zi定义为三个主成分

至此，PCA结束，可见，PCA只适用于初步建立模型时，降低变量的工作，把一些影响小的因子去除



然后再把建立好的模型推广到 worldwide 过程中，在进行优化（这一步，大多与聚类分析结合）。





