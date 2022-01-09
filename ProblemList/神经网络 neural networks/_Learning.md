# Neural Network

​	这是一个非常大的主题，或者说模型。

​	如果能够运用其中的某一类解决一些**很难解析**的问题，那么会大大减少我们解决问题的难度。

​	首先对这个非常大的话题分类，然后再对各种类型一一讨论。

## Types of Neural Network


链接：https://zhuanlan.zhihu.com/p/372516381



### 十一大必知网络结构

#### **1.Perceptron**

感知机是所有神经网络的基础，主要由全连接层组成，下面是感知机示意图。

![img](https://pic4.zhimg.com/v2-eb329263aba0f91b2361ecb021d8802b_b.jpg)

#### **2.Feed-Forward Network(FNN)**

FNN是有Perceptron组合得到的，由输入层、隐藏层和输出层组成，其结构如下：

![img](https://pic1.zhimg.com/v2-e024cdd12d4bd99effa8abd43fd41aa8_b.jpg)

#### **3.Residual Networks (ResNet)**

[深度神经网络](https://www.zhihu.com/search?q=深度神经网络&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"372516381"})最大问题就是当网络深度达到一定程度时就会出现消失梯度的问题，导致模型训练不佳，为了缓解该问题，我们设计了[残差网络](https://www.zhihu.com/search?q=残差网络&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"372516381"})。它通过一个“跳跃”层传播信息号，大大缓解了梯度消失的问题。

![img](https://pic1.zhimg.com/v2-a678ca93ec3e3cdef9e66927e5db8414_b.jpg)

#### **4.Recurrent Neural Network (RNN)**

[递归神经网络](https://www.zhihu.com/search?q=递归神经网络&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"372516381"})是早期处理序列问题的始祖，它包含循环，并在自身上进行递归，因此得名“递归”。RNN允许将信息存储在网络中，并使用先前训练中的推理，对即将发生的事件做出更好、更明智的决策。

![img](https://pic1.zhimg.com/v2-9b59c0ebcebdf93d75475dae29ce222c_b.jpg)

#### **5.Long Short Term Memory Network (LSTM)**

RNN最大的问题在于，一旦处理的序列较长的时候，例如100，RNN的效果就会大大变差，所以大家设计了LSTM，LSTM可以处理大约300左右长度的序列，这也是为什么目前LSTM在序列化的问题中还经常被使用的原因。

![img](https://pic3.zhimg.com/v2-aa66adc02aa0cac33f909fba45565e0a_b.jpg)

#### **6.Echo State Networks(ESN)**

[回声状态网络](https://www.zhihu.com/search?q=回声状态网络&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"372516381"})由输入层、隐藏层(即储备池)、输出层组成，是递归神经网络的一种变体，它有一个非常稀疏连接的隐层（通常是百分之一左右的连通性）。[神经元](https://www.zhihu.com/search?q=神经元&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"372516381"})的连接和权值是随机分配的，忽略层和神经元的差异（跳跃连接）。ESN将隐藏层设计成一个具有很多神经元组成的[稀疏网络](https://www.zhihu.com/search?q=稀疏网络&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"372516381"})，通过调整网络内部权值的特性达到记忆数据的功能。

![img](https://pic4.zhimg.com/v2-8b1509cc59f2866a9d0ff1814c82ddbf_b.jpg)

#### **7.Convolutional Neural Network (CNN)**

CNN早期在图像中取得了巨大的成功，在今日，CNN仍然是不可或缺的一部分。因为图像数据有着非常高的维度，训练一个标准网络训练图像（例如简单的MLP）都需要数十万个输入神经元，除了明显的高计算开销外，还会导致许多与神经网络维数灾难相关的问题。CNN则利用卷积层来帮助降低图像的维数，不仅大大降低了训练的参数，而且在效果上也取得了巨大的提升。

![img](https://pic3.zhimg.com/v2-edbcf67ab1ced9fe4c49b82fb7a7351a_b.jpg)

#### **8.Deconvolutional Neural Network (DNN)**

反卷积神经网络，顾名思义，其性能与卷积神经网络相反。DNN并不是通过卷积来降低来图像的维数，而是利用反卷积来创建图像，一般是从噪声中生成的。DNN还经常用于寻找丢失的特征或信号，这些特征或信号以前可能被认为对[卷积神经网络](https://www.zhihu.com/search?q=卷积神经网络&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"372516381"})的任务并不重要。一个信号可能由于与其他信号卷积而丢失。信号的Deconvolution可以用于图像合成和分析。

![img](https://pic1.zhimg.com/v2-6f51d07f73b61aa2120a98ed76d2b368_b.jpg)

#### **9.Generative Adversarial Network (GAN)**

生成性对抗网络是一种专门设计用来生成图像的网络，它由两个网络组成：一个生成器和一个判别器。判别器的任务是区分图像是从数据集中提取的还是由生成器生成的，生成器的任务是生成虚假的图像，尽可能使得判别器无法区分图像是否真实，目前GAN生成的图像很多都是栩栩如生，甚至达到了以假乱真的程度。

![img](https://pic2.zhimg.com/v2-ca2590fa64e50e019472cfdd2c6558dd_b.jpg)

#### **10.Auto Encoder (AE)**

自动编码器的应用非常广，包括模型压缩、数据去噪、异常检测、推荐系统等等。其基本思想是将原始的高维数据“压缩”、低维的数据，然后将压缩后的数据投影到一个新的空间中。

![img](https://pic4.zhimg.com/v2-2021298d5264a2ecc9bc8a43dde0e423_b.jpg)

#### **11.Variational Auto Encoder (VAE)**

自动编码器学习输入的压缩表示，而变分自动编码器（VAE）学习表示数据的概率分布的参数。它不只是学习表示数据的函数，而是获得更详细和细致的数据视图，从分布中采样并生成新的输入数据样本。所以VAE更像是一个“生成”模式，类似于GAN。

![img](https://pic1.zhimg.com/v2-a312847d56ec351735ae8c1f900662d0_b.jpg)

## Basics

### Two-Layer Neural Network(基本就是`zhihu`然后我加了点重点和改了下格式)



---



链接可能失效，但是以下文件包括所有信息，且格式调整并突出重点以及增加批注。



作者：Mr.看海
链接：https://zhuanlan.zhihu.com/p/65472471
来源：知乎

---

#### 1.简化的两层神经网络**使用**

首先去掉图1中一些难懂的东西，如下图（请仔细看一下图中的标注）：

![img](https://pic2.zhimg.com/v2-7ee8cabcbd707dd4deab7155af2ba4cd_b.jpg)图2.简化过后的两层神经网络

**1.1.输入层**

在我们的例子中，输入层是[坐标值](https://www.zhihu.com/search?q=坐标值&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"65472471"})，例如（1,1），这是一个包含两个元素的数组，也可以看作是一个1*2的矩阵。输入层的元素维度与输入量的特征息息相关，如果输入的是一张32*32像素的灰度图像，那么输入层的维度就是32*32。

**1.2.从输入层到隐藏层**

连接输入层和隐藏层的是W1和b1。由X计算得到H十分简单，就是矩阵运算：

![img](https://pic2.zhimg.com/v2-b31ecd1eea01a5e52968075778cb9699_b.png)

如果你学过线性代数，对这个式子一定不陌生。如上图中所示，在设定隐藏层为50维（也可以理解成50个神经元）之后，矩阵H的大小为（1*50）的矩阵。

**1.3.从隐藏层到输出层**

连接隐藏层和输出层的是W2和b2。同样是通过矩阵运算进行的：

![img](https://pic4.zhimg.com/v2-0c8c9f5ea2376623cb31ba74e9256627_b.png)

**1.4.分析**

通过上述两个线性方程的计算，我们就能得到最终的输出Y了，但是如果你还对线性代数的计算有印象的话，应该会知道：***一系列线性方程的运算最终都可以用一个线性方程表示\***。也就是说，上述两个式子联立后可以用一个线性方程表达。对于两次神经网络是这样，就算网络深度加到100层，也依然是这样。这样的话神经网络就失去了意义。

所以这里要对网络注入灵魂：**激活层**。



#### 2.激活层

简而言之，激活层是为矩阵运算的结果添加非线性的。常用的激活函数有三种，分别是[阶跃函数](https://www.zhihu.com/search?q=阶跃函数&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"65472471"})、Sigmoid和ReLU。不要被奇怪的函数名吓到，其实它们的形式都很简单，如下图：

![img](https://pic2.zhimg.com/v2-5600c3448f3cb260702e7460cfb0be31_b.jpg)

>图3.三种常用的激活函数

* `阶跃函数`：当输入小于等于0时，输出0；当输入大于0时，输出1。

* `Sigmoid`：当输入趋近于正无穷/负无穷时，输出无限接近于1/0。

* `ReLU`：当输入小于0时，输出0；当输入大于0时，输出等于输入。



其中，阶跃函数输出值是跳变的，且只有二值，较少使用；Sigmoid函数在当x的[绝对值](https://www.zhihu.com/search?q=绝对值&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"65472471"})较大时，曲线的斜率变化很小（梯度消失），并且计算较复杂；`ReLU`是当前较为常用的激活函数。

激活函数具体是怎么计算的呢？

假如经过公式**H=X\*W1+b1**计算得到的H值为：$(1,-2,3,-4,7,\dots)$，那么经过阶跃函数激活层后就会变为$(1,0,1,0,1,\dots)$，经过`ReLU`激活层之后会变为$(1,0,3,0,7,\dots)$。





**需要注意的是，每个隐藏层计算（矩阵线性运算）之后，都需要加一层激活层，要不然该层线性计算是没有意义的。**





此时的神经网络变成了如下图所示的形式：

![img](https://pic1.zhimg.com/v2-0ced86f32dfa241fc9de10421edbd9b4_b.jpg)

> 图4.加上激活层的两层神经网络

我们都知道神经网络是分为“训练”和“使用”两个步骤的。如果是在“使用”的步骤，图4就已经完成整个过程了，在求得的Y（大小为1*4）矩阵中，数值最大的就代表着当前分类。

但是对于用于“训练”的网络，图4还远远不够。起码当前的输出Y，还不够“漂亮”。

#### 3.输出的正规化

在图4中，输出Y的值可能会是$(3,1,0.1,0.5)$这样的矩阵，诚然我们可以找到里边的最大值“3”，从而找到对应的分类为I，但是这并不直观。我们想让最终的输出为**概率**，也就是说可以生成像$(90\%,5\%,2\%,3\%)$这样的结果，这样做不仅可以找到最大概率的分类，而且可以知道各个分类计算的概率值。



```text
这里Y的数值大小代表了可能性，为什么是可能性呢？因为这是我们定义的，我们通过定义可能性来进行负反馈调节，让可能性最符合我们训练的数据。所谓正规化指将可能性转换成百分比，取指数是考虑到负数。
```



具体是怎么计算的呢？

计算公式如下：

![img](https://pic4.zhimg.com/v2-3ad93ae576918ff385485dab6a2e6b87_b.png)

简单来说分三步进行：

1. 以e为底对所有元素求指数幂；
2. 将所有指数幂求和；
3. 分别将这些指数幂与该和做商。

这样求出的结果中，所有元素的和一定为1，而每个元素可以代表概率值。

我们将使用这个计算公式做输出结果正规化处理的层叫做`Softmax`层。此时的神经网络将变成如下图所示：

![img](https://pic2.zhimg.com/v2-01285f87ff9d523f62d2d4f6586583c5_b.jpg)

>  图5.输出正规化之后的神经网络

#### 4.如何衡量输出的好坏

通过`Softmax`层之后，我们得到了`I，II，III和IV`这四个类别分别对应的概率，但是要注意，这是神经网络计算得到的概率值结果，而非真实的情况。

比如，`Softmax`输出的结果是$(90\%,5\%,2\%,3\%)$，真实的结果是$(100\%,0,0,0)$。虽然输出的结果可以正确分类，但是与真实结果之间是有差距的，一个优秀的网络对结果的预测要无限接近于100%，为此，我们需要将`Softmax`输出结果的好坏程度做一个“量化”。

一种直观的解决方法，是用1减去`Softmax`输出的概率，比如1-90%=0.1。不过更为常用且巧妙的方法是，求**对数的负数**。

还是用90%举例，对数的负数就是：-log0.9=0.046

可以想见，概率越接近100%，该计算结果值越接近于0，说明结果越准确，该输出叫做“**交叉熵损失**（Cross Entropy Error）”。

我们训练神经网络的目的，就是尽可能地减少这个“交叉熵损失”。

此时的网络如下图：

![img](https://pic3.zhimg.com/v2-55f56e273500c8881440877d9c43ebba_b.jpg)图6.计算交叉熵损失后的神经网络

#### 5.反向传播与参数优化

上边的1~4节，讲述了神经网络的正向传播过程。一句话复习一下：**神经网络的传播都是形如Y=WX+b的矩阵运算；为了给矩阵运算加入非线性，需要在隐藏层中加入激活层；输出层结果需要经过`Softmax`层处理为概率值，并通过交叉熵损失来量化当前网络的优劣。**

算出交叉熵损失后，就要开始反向传播了。其实反向传播就是一个**参数优化**的过程，优化对象就是网络中的所有W和b（因为其他所有参数都是确定的）。

神经网络的神奇之处，就在于它可以自动做W和b的优化，在深度学习中，参数的数量有时会上亿，不过其优化的原理和我们这个两层神经网络是一样的。

这里举一个形象的例子描述一下这个参数优化的原理和过程：

假设我们操纵着一个球型机器行走在沙漠中

![img](https://pic2.zhimg.com/v2-ce4acebca3fecaf429a077e16ff989d9_b.jpg)

我们在机器中操纵着四个旋钮，分别叫做W1，b1，W2，b2。当我们旋转其中的某个旋钮时，球形机器会发生移动，但是旋转旋钮大小和机器运动方向之间的对应关系是不知道的。而我们的目的就是**走到沙漠的最低点**。

![img](https://pic4.zhimg.com/v2-4dfad8b96d10df776afdcaa618d59857_b.jpg)

此时我们该怎么办？只能挨个试喽。

* 如果增大W1后，球向上走了，那就减小W1。
* 如果增大b1后，球向下走了，那就继续增大b1。
* 如果增大W2后，球向下走了一大截，那就多增大些W2。

> https://zhuanlan.zhihu.com/p/66534632

然后花点时间看下上面这篇文章

#### 6.迭代

神经网络需要反复迭代。

 如上述例子中，第一次计算得到的概率是90%，交叉熵损失值是0.046；将该损失值反向传播，使W1,b1,W2,b2做相应微调；再做第二次运算，此时的概率可能就会提高到92%，相应地，损失值也会下降，然后再反向传播损失值，微调参数W1,b1,W2,b2。依次类推，损失值越来越小，直到我们满意为止。

 此时我们就得到了理想的W1,b1,W2,b2。

此时如果将任意一组坐标作为输入，利用图4或图5的流程，就能得到分类结果。

### Implementation

然后学习一个经典案例就是训练分点，即给定$(x,y)$，程序能分辨此点在第$k$象限。

#### Prerequisites

* `Numpy` 的一些细节
  * shape[0] ：第一维长度
  * ![image-20220108195100024](https://s2.loli.net/2022/01/08/x8dEXtacjDILV7l.png)
    * 注意观察-1的情况
* code as follow

```python
# 作者：Mr.看海
# 链接：https://zhuanlan.zhihu.com/p/67682601
# 来源：知乎
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

import numpy as np  
def affine_forward(x, w, b):   
    out = None                       # 初始化返回值为None
    N = x.shape[0]                   # 重置输入参数X的形状
    x_row = x.reshape(N, -1)         # (N,D)
    out = np.dot(x_row, w) + b       # (N,M)
    cache = (x, w, b)                # 缓存值，反向传播时使用
    return out,cache

def affine_backward(dout, cache):   
    x, w, b = cache                              # 读取缓存
    dx, dw, db = None, None, None                # 返回值初始化
    dx = np.dot(dout, w.T)                       # (N,D)    
    dx = np.reshape(dx, x.shape)                 # (N,d1,...,d_k)   
    x_row = x.reshape(x.shape[0], -1)            # (N,D)    
    dw = np.dot(x_row.T, dout)                   # (D,M)    
    db = np.sum(dout, axis=0, keepdims=True)     # (1,M)    
    return dx, dw, db


X = np.array([[2,1],  
            [-1,1],  
            [-1,-1],  
            [1,-1]])      # 用于训练的坐标，对应的是I、II、III、IV象限
t = np.array([0,1,2,3])   # 标签，对应的是I、II、III、IV象限
np.random.seed(1)         # 有这行语句，你们生成的随机数就和我一样了

# 一些初始化参数  
input_dim = X.shape[1]     # 输入参数的维度，此处为2，即每个坐标用两个数表示
num_classes = t.shape[0]   # 输出参数的维度，此处为4，即最终分为四个象限
hidden_dim = 50            # 隐藏层维度，为可调参数
reg = 0.001                # 正则化强度，为可调参数
epsilon = 0.001            # 梯度下降的学习率，为可调参数
# 初始化W1，W2，b1，b2
W1 = np.random.randn(input_dim, hidden_dim)     # (2,50)
W2 = np.random.randn(hidden_dim, num_classes)   # (50,4)
b1 = np.zeros((1, hidden_dim))                  # (1,50)
b2 = np.zeros((1, num_classes))                 # (1,4)



for j in range(10000):   #这里设置了训练的循环次数为10000
 # ①前向传播
    H,fc_cache = affine_forward(X,W1,b1)                 # 第一层前向传播
    H = np.maximum(0, H)                                 # 激活
    relu_cache = H                                       # 缓存第一层激活后的结果
    Y,cachey = affine_forward(H,W2,b2)                   # 第二层前向传播        
 # ②Softmax层计算
    probs = np.exp(Y - np.max(Y, axis=1, keepdims=True))    
    probs /= np.sum(probs, axis=1, keepdims=True)        # Softmax算法实现
 # ③计算loss值
    N = Y.shape[0]                                       # 值为4
    print(probs[np.arange(N), t])                        # 打印各个数据的正确解标签对应的神经网络的输出
    loss = -np.sum(np.log(probs[np.arange(N), t])) / N   # 计算loss
    print(loss)                                          # 打印loss
 # ④反向传播
    dx = probs.copy()                                    # 以Softmax输出结果作为反向输出的起点
    dx[np.arange(N), t] -= 1                             # 
    dx /= N                                              # 到这里是反向传播到softmax前
    dh1, dW2, db2 = affine_backward(dx, cachey)          # 反向传播至第二层前
    dh1[relu_cache <= 0] = 0                             # 反向传播至激活层前
    dX, dW1, db1 = affine_backward(dh1, fc_cache)        # 反向传播至第一层前
# ⑤参数更新
    dW2 += reg * W2
    dW1 += reg * W1
    W2 += -epsilon * dW2
    b2 += -epsilon * db2
    W1 += -epsilon * dW1
    b1 += -epsilon * db1
```



## Why do we need multiple layers?

> 一层的神经网络是理论上已经被证明能够表示任何函数
>
> 也就是说
>
> **无论什么问题，都能被一层神经网络训练得到答案**。

这是一个非常恐怖的事实，也就是说

* 我们之前解决的**所有问题**，无论是什么领域的，都是给定条件，找出答案
  * 条件，就是自变量，答案就是因变量
* 也就是说 所有问题都能通过一层神经网络训练实现。

但是

* 隐藏层的节点数是无法确定的

考虑函数

* $f(x)=rand(x)$

这里隐藏层的节点数是$\infin$,

所以理论上能表示，并不代表你能在有限的时间内实现

所以需要根据需要，增加层数，

* 实质上就是增加非线性，可以更好地表现出**信息间交互的特征**

  

相关帖子,如下，是**[David Torpey](https://www.quora.com/profile/David-Torpey-2)**写的

> Neural networks (kind of) need multiple layers in order to learn more detailed and more abstractions relationships within the data and how the features interact with each other on a non-linear level.
>
> Even though it is theoretically possible to represent any possible function with a single hidden layer neural network, determine the number of nodes needed in that hidden layer is difficult. Therefore, adding more layers (apart from increasing computational complexity to the training and testing phases), allows for more easy representation of the interactions within the input data, as well as allows for more abstract features to be learned and used as input into the next hidden layer.
>
> Regression example: Learning to predict the selling price of houses given features: number of rooms, number of bathrooms, number of windows, size of house in ft^2, distance from highways, age of building, etc.
>
> The first layer might learn simple, non-abstract features about how these input features relate to the selling price, such as: more rooms or a bigger house = higher selling prices and vice versa.
>
> As we get deeper into the network, the features it will have learned will be much more complex and much more abstract, such as: small house + very old house = high selling price and big house + small number of bathrooms = low selling prices. The point is that the features learned in the deeper layers get much more intricate and detailed, and the network learns how these abstract features affect the selling price.
>
> Classification example: Classify objects in images
>
> We feed in raw images (the pixels values) into a convolutional neural network with many layers. The first layer might learn simple geometrical objects such as lines that signify the object we are trying to classify. The deeper layers will learn much more abstract, detailed features about the objects such as sets of lines that define shapes, and then eventually sets of these shapes from the earlier layers that make up the shape of the object we are trying to classify, for example a car.
>
> Deep is essentially features learning. This is why you need the network to have many layers (i.e. deep). They need to have many layers of abstraction since we want the neural network to learn as well as possible what type of non-linear manifold in the high dimensions the input data lies on.



## Deeper(Real) Usage

* 基于CS224b Stanford University 所谓入门 ~~我没看~~ ~~真的不想看qwq~~
  * https://www.zybuluo.com/hanbingtao/note/433855

* https://www.youtube.com/watch?v=tp3KIQx4Mkk&ab_channel=NuruzzamanFaruqui
