# Cellular Automaton

## What is Cellular Automation?

作者：交通科研Lab
链接：https://www.zhihu.com/question/21929655/answer/919048716
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



**一下很多是视频，可以点下就能播放，这应该是最生动的一个帖子了**



> **元胞自动机**（Cellular Automata，CA）是20世纪50年代初由计算机之父冯·诺依曼（J.von Neumann）为了模拟[生命系统](https://www.zhihu.com/search?q=生命系统&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A919048716})所具有的自复制功能而提出来的。此后，[史蒂芬·沃尔夫勒姆](https://www.zhihu.com/search?q=史蒂芬·沃尔夫勒姆&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A919048716})（Stephen Wolfram）对元胞自动机理论进行了深入的研究。例如，他对一维初等元胞机全部256种规则所产生的模型进行了深入研究，并将元胞自动机分为**平稳型**、**周期型**、**混沌型**和**复杂型** 4 种类型。



- 举个简单的例子，这是包含800个时间步的[90号规则](https://www.zhihu.com/search?q=90号规则&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A919048716})演化图案。虽然初始状态仅仅是一个点，但是随着时间的推移，简单的演化规则使图案分形，并在空间上形成了嵌套的三角形状。



![img](https://pic3.zhimg.com/50/v2-fcf7e76adc1ea7ed66bbbf36e40eb411_720w.jpg?source=1940ef5c)![img](https://pic3.zhimg.com/80/v2-fcf7e76adc1ea7ed66bbbf36e40eb411_1440w.jpg?source=1940ef5c)



**那么，元胞自动机到底是什么呢？**

简单来讲，

元胞自动机就是

采用离散的空间布局和时间间隔，



![img](https://pic2.zhimg.com/50/v2-9a1e0ac9b4cfc3b3568259872e0c76ff_720w.jpg?source=1940ef5c)![img](https://pic2.zhimg.com/80/v2-9a1e0ac9b4cfc3b3568259872e0c76ff_1440w.jpg?source=1940ef5c)



将[元胞](https://www.zhihu.com/search?q=元胞&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A919048716})分成有限种状态，



![img](https://pic3.zhimg.com/50/v2-57eb2ffdfab50bfa295a96b46bd2e97e_720w.jpg?source=1940ef5c)![img](https://pic3.zhimg.com/80/v2-57eb2ffdfab50bfa295a96b46bd2e97e_1440w.jpg?source=1940ef5c)



元胞个体状态的演变，仅与其当前状态以及其某个局部邻域的状态有关。



![img](https://pic3.zhimg.com/50/v2-e01c0116b8318a8bf47628698e5b0180_720w.jpg?source=1940ef5c)![img](https://pic3.zhimg.com/80/v2-e01c0116b8318a8bf47628698e5b0180_1440w.jpg?source=1940ef5c)



元胞自动机以计算机建模和仿真的方法，研究类似于生物细胞（cell）的，由大量并行单元个体组成的复杂系统的宏观行为与规律。L-系统、格子气模型、格子气-Boltzmann方法、交通流模型等都是元胞自动机的具体化，有着重要的理论意义和实际应用价值。



**元胞自动机的基本要素如下：**

- **空间：**元胞在空间中分布的空间格点，可以是一维、二维或多维。
- **状态集：**可以是两种状态，用“生”、“死”，“0”、“1”，“黑”、“白”来表示；也可以是多种状态，如不同的颜色。
- **邻居：**存在与某一元胞周围，能影响该元胞在下一时刻的状态。
- **演化规则：**根据元胞及其邻居元胞的状态，决定下一时刻该元胞状态的[动力学函数](https://www.zhihu.com/search?q=动力学函数&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A919048716})，也可以是状态转移方程。





- 这是一个算法生成的三维空间中的元胞自动机：



![img](https://pic2.zhimg.com/50/v2-e10ce57a012d1f3cefc21956c2ee68d9_720w.jpg?source=1940ef5c)![img](https://pic2.zhimg.com/80/v2-e10ce57a012d1f3cefc21956c2ee68d9_1440w.jpg?source=1940ef5c)





- 还有曾经火遍全世界的生命游戏



![img](https://pic3.zhimg.com/50/v2-1d24b7991aab26549c466a2961b7181f_720w.jpg?source=1940ef5c)![img](https://pic3.zhimg.com/80/v2-1d24b7991aab26549c466a2961b7181f_1440w.jpg?source=1940ef5c)



> 生命游戏（Game of Life），或者叫它的全称John Conway's Game of Life。是英国数学家[约翰·康威](https://www.zhihu.com/search?q=约翰·康威&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A919048716})在1970年代所发明的一种元胞自动机。

在二维平面上的方格细胞里，每个细胞有两种状态：死或活，而下一回合的状态完全受它周围8个细胞的状态而定。按照以下三条规则进行演化：

\1. 活细胞周围的细胞数小于2个或多于3个则死亡； 

\2. 活细胞周围有2或3个细胞可以继续存活； 

\3. 死细胞周围恰好有3个细胞则会复活。



生命游戏虽然看起来简单，但经大佬们操作，真可谓出神入化！



![img](https://pic2.zhimg.com/50/v2-9a9835122f209cc9e2bc38e2de88f613_720w.jpg?source=1940ef5c)



明湖鹅们都震惊了！

可周期循环

![img](https://pic1.zhimg.com/50/v2-9ea079969a5383458bde042557b5a4f5_720w.jpg?source=1940ef5c)

可腾挪平移

<video class="ztext-gif GifPlayer-gif2mp4" src="https://vdn.vzuu.com/SD/5c10d214-235b-11eb-97a7-d2f41dd98737.mp4?disable_local_cache=1&amp;auth_key=1641624651-0-0-17b12fec8f868a3768b32315195c2d34&amp;f=mp4&amp;bu=pico&amp;expiration=1641624651&amp;v=ali" data-thumbnail="https://pica.zhimg.com/50/v2-5eb5244648ce5f773588afc125e7550d_720w.jpg?source=1940ef5c" poster="https://pica.zhimg.com/50/v2-5eb5244648ce5f773588afc125e7550d_720w.jpg?source=1940ef5c" data-size="normal" preload="metadata" loop="" playsinline=""></video>



可简单如宇宙飞船

<video class="ztext-gif GifPlayer-gif2mp4" src="https://vdn3.vzuu.com/SD/e5010d22-ec84-11ea-acfd-5ab503a75443.mp4?disable_local_cache=1&amp;auth_key=1641624651-0-0-01d77f058592b2b287649324690ab62c&amp;f=mp4&amp;bu=pico&amp;expiration=1641624651&amp;v=tx" data-thumbnail="https://pica.zhimg.com/50/v2-f7735804f2b305bea691d0365b5731e7_720w.jpg?source=1940ef5c" poster="https://pica.zhimg.com/50/v2-f7735804f2b305bea691d0365b5731e7_720w.jpg?source=1940ef5c" data-size="normal" preload="metadata" loop="" playsinline=""></video>



可也复杂如模拟时钟

<video class="ztext-gif GifPlayer-gif2mp4" src="https://vdn3.vzuu.com/SD/a768d1e4-1f55-11eb-8c57-667268985f5c.mp4?disable_local_cache=1&amp;auth_key=1641624651-0-0-2de039dc57ad00ff4c8c6e8d0ff012c7&amp;f=mp4&amp;bu=pico&amp;expiration=1641624651&amp;v=tx" data-thumbnail="https://pic1.zhimg.com/50/v2-02bc11a92522bd3268c34e878064d85e_720w.jpg?source=1940ef5c" poster="https://pic1.zhimg.com/50/v2-02bc11a92522bd3268c34e878064d85e_720w.jpg?source=1940ef5c" data-size="normal" preload="metadata" loop="" playsinline=""></video>



------



言归正传，让我们来谈谈**元胞自动机在交通领域的应用**。元胞自动机在我们交通领域应用非常广泛，常常被用来模拟道路上的车辆或移动的行人。

话不多说，直接上图。

- 这是经典的NaSch模型，模拟了车辆在一维高速公路上的行驶状态

![img](https://pic1.zhimg.com/50/v2-b2257119ede75ba235cbd938d0736004_720w.jpg?source=1940ef5c)

- 将CA模型推广到二维，可以仿真换道的行人流或车流

![img](https://pic3.zhimg.com/50/v2-8f6ba0eead51c46ed921d33884c1cbf6_720w.jpg?source=1940ef5c)

<video class="ztext-gif GifPlayer-gif2mp4" src="https://vdn1.vzuu.com/SD/5b6d551e-2381-11eb-87f1-faee0cc0063c.mp4?disable_local_cache=1&amp;auth_key=1641624653-0-0-c53143a0cfad1c4adda0c3df105ec548&amp;f=mp4&amp;bu=pico&amp;expiration=1641624653&amp;v=hw" data-thumbnail="https://pic3.zhimg.com/50/v2-8f6ba0eead51c46ed921d33884c1cbf6_720w.jpg?source=1940ef5c" poster="https://pic3.zhimg.com/50/v2-8f6ba0eead51c46ed921d33884c1cbf6_720w.jpg?source=1940ef5c" data-size="normal" preload="metadata" loop="" playsinline=""></video>



- 通过设置更复杂的规则，可以用元胞自动机仿真更真实的情况。例如在对向行走的行人流情境下，元胞们会仿真行人的排队特性，从而避免冲突相撞。

![img](https://pic2.zhimg.com/50/v2-332188684ca16eb388779eb1af2789a9_720w.jpg?source=1940ef5c)

<video class="ztext-gif GifPlayer-gif2mp4" src="https://vdn1.vzuu.com/SD/e14c45c6-2338-11eb-9683-26c0f30bdd89.mp4?disable_local_cache=1&amp;auth_key=1641624653-0-0-2beea2b23b9c606bb8df69efa171a53d&amp;f=mp4&amp;bu=pico&amp;expiration=1641624653&amp;v=hw" data-thumbnail="https://pic2.zhimg.com/50/v2-332188684ca16eb388779eb1af2789a9_720w.jpg?source=1940ef5c" poster="https://pic2.zhimg.com/50/v2-332188684ca16eb388779eb1af2789a9_720w.jpg?source=1940ef5c" data-size="normal" preload="metadata" loop="" playsinline=""></video>







看到这里是不是有一点点小心动呢？

你也想编一个元胞自动机吗？



### 我们以`NaSch`模型为例，用Python实现



- 老规矩，先调包，并创建一个图像

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
# 创建图像
fig = plt.figure(figsize=(10,1))
```



- 设置模型的参数，相当于一个遥控器，哪里不对点哪里~

```python
# 模型参数设置
numofcell = 20     # 道路长度
numofcar = 12      # 空间中的车辆数
max_time = 100     # 设置最大时间步
max_speed = 5      # 允许的车辆最大速度
p_slowdown = 0.3   # 随机慢化概率
pause_time = 0.1   # 刷新时间（每一帧持续的时间）
cell_size = 15     # 元胞的大小
```



- 定义一个函数来构建一维空间，空间的长度就是道路长度，相当于用一系列和X轴或Y轴平行的直线，绘制一排小网格，每个小网格的中心，相当于(i,0)，其中，i=1,2,…,numofcell

```python
# 函数：构建一维空间
def Plot_Space():
    for i in range(1, numofcell): plt.plot([i-0.5, i-0.5], [-0.5, 0.5], '-k', linewidth = 0.5)       
    plt.axis([-0.5, numofcell-0.5, -0.5, 0.5])
plt.xticks([]);plt.yticks([])
```



![img](https://pic1.zhimg.com/50/v2-1d7a0167f9c813df44a37cd317f86592_720w.jpg?source=1940ef5c)![img](https://pic1.zhimg.com/80/v2-1d7a0167f9c813df44a37cd317f86592_1440w.jpg?source=1940ef5c)





- 定义一个函数来获取和前车的距离，从而避免两车相撞。在这里采用了[周期性边界](https://www.zhihu.com/search?q=周期性边界&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A919048716})，即从道路末端驶出的小车会重新回到起点，相当于一个环路。

```python
# 函数：获取和前车的距离
def get_empty_front(link, numofcell, indexofcell):
    link2 = link * 2   # 周期性边界
    num = 0; i = 1
    while (link2[indexofcell + i]) == None:
        num += 1; i += 1
    return num
```



- 在道路空间中随机生成numofcar个[初始元胞](https://www.zhihu.com/search?q=初始元胞&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A919048716})，并赋予随机的初始速度（不大于已经设置好的最大速度）。道路被车辆占有的状态储存在列表link中，若元胞中没有车辆，则link对应的位置为“None”；若元胞中有车，link对应的位置储存车辆的速度。（可以开开脑洞，大胆地尝试不同初始状态噢）

```python
# 随机生成初始元胞
Plot_Space()
link = [None] * numofcell
num = 0
while num != numofcar:
    sj = random.randint(0, numofcell - 1)
    if link[sj] == None:
        link[sj] = random.randint(0, 5)
        num += 1
```



- 在0~max_time的时间步内，进行NaSch模型的演化，模型演化的四个步骤：

**加速：**若还没到速度最大值，速度就加1

**减速：**如果速度值大于前方的空元胞数，有撞车风险，则减速至前方的空元胞数目

**随机慢化：**司机以p_slowdown的概率随机踩刹车，使速度减1

**位置更新：**新的位置为当前位置+当前速度，同时更新所有车辆的位置

```python
# NaSch模型
for t in range(0, max_time):
    for cell in range(0, numofcell):
        if link[cell] != None:
            # 加速
            link[cell] = min(link[cell] + 1, max_speed)
            # 减速
            link[cell] = min(link[cell], get_empty_front(link, numofcell, cell))
            # 随机慢化
            if random.random() <= p_slowdown:
                link[cell] = max(link[cell] - 1, 0)
    # 位置更新
    nlink = [None] * numofcell
    for cell in range(0, numofcell):
        if link[cell] != None:
            new_index = cell + link[cell]
            if new_index >= numofcell:
                new_index -= numofcell
            nlink[new_index] = link[cell]
    link = nlink
```



- 绘制当前时间步车辆位置的图像，注意这里有缩进，说明上一段NaSch模型演化代码中的for循环还没有结束呦~

```python
x1 = []
    for i in range(0,len(link)):
        if link[i] != None:
            x1.append(i)
    Plot_Space()
    plt.plot(x1, [0] * numofcar, 'sk', markersize=cell_size)
    plt.xlabel('timestep:' + str(t))
```



- 让图片动起来（**同样注意缩进**），plt.pause()函数让当前的图像维持pause_time长的时间，随后plt.cla()函数将整个图像擦除。下一个时间步，继续绘制图像，保留一段时间，并擦除，再绘制图像……就能源源不断地产生动画啦！你学会了吗？

```python
    plt.pause(pause_time)
    plt.cla()
```



**终于写完代码啦，RUN一下！**

仔细看，拥堵带也会跳鬼步舞，逐渐向后传播的。

<video class="ztext-gif GifPlayer-gif2mp4" src="https://vdn3.vzuu.com/SD/ffc0cac6-233e-11eb-91c8-ba4b3ea4a575.mp4?disable_local_cache=1&amp;auth_key=1641624655-0-0-2f6f312edafa248f430e910a52839c9c&amp;f=mp4&amp;bu=pico&amp;expiration=1641624655&amp;v=tx" data-thumbnail="https://pic2.zhimg.com/50/v2-3b6878489f851d07be894b5fb569886d_720w.jpg?source=1940ef5c" poster="https://pic2.zhimg.com/50/v2-3b6878489f851d07be894b5fb569886d_720w.jpg?source=1940ef5c" data-size="normal" preload="metadata" loop="" playsinline=""></video>



操控我的“遥控器”，给它一个初始状态，就可以开启“阅兵模式”啦，你猜我是怎么做到的？

![img](https://pic3.zhimg.com/50/v2-c6155fd05a1d22975edbb7f376ee213a_720w.jpg?source=1940ef5c)

给我开“瞬移”挂，只要动的足够快，你就分不清我到底是往左跑还是往右跑

<video class="ztext-gif GifPlayer-gif2mp4" src="https://vdn3.vzuu.com/SD/270fb522-2369-11eb-9f4b-b6d50edeb516.mp4?disable_local_cache=1&amp;auth_key=1641624655-0-0-5d538177319f78909c49c512c308357c&amp;f=mp4&amp;bu=pico&amp;expiration=1641624655&amp;v=tx" data-thumbnail="https://pica.zhimg.com/50/v2-6f24cdd31dac3c45f1f70c8de000295d_720w.jpg?source=1940ef5c" poster="https://pica.zhimg.com/50/v2-6f24cdd31dac3c45f1f70c8de000295d_720w.jpg?source=1940ef5c" data-size="normal" preload="metadata" loop="" playsinline=""></video>

## Related Topics

### Game of Life (Weilie hw)

​	没有任何应用价值，但这个自动机就是这个游戏表达的意思。

### Traffic (2014 Problem A) 裸题

* 把道路（连续的）给离散成2维平面上的点，每个点可以是0/1，也就是有车/没有车
* 每个时间点整个道路是一个状态$S$
* $S$到下个时间点$S'$会进行一次转移，也就是我们所学的$FSM$的那种转移
* 不断的进行转移，就模拟了交通
* $S\rightarrow S'$ 所用的函数，（如果线性的话相当于直接乘一个矩阵），是一个**转移的方程**。



## 好处

* 不用推公式
* 设定转移方程/转移函数 十分感性，不需要严谨的推导，甚至可以试出来
  * 比如`NaSch` 的转移如下
    * ![image-20220108180049022](https://s2.loli.net/2022/01/08/XcBDpZyrInJ4zsv.png)
* 可以研究难易找到研究对象的问题
  * 比如连续性问题
  * 比如对象非常多，难以一一研究的问题
* 应用
  * ![img](https://images2017.cnblogs.com/blog/748759/201712/748759-20171220180928100-2122491144.png)



## 相关文献

https://wenku.baidu.com/view/76c47b2c42323968011ca300a6c30c225901f001.html

https://www.zhihu.com/question/21929655

https://www.cnblogs.com/bellkosmos/p/introduction_of_cellular_automata.html （这个很好）

## 实质

* 后来想了下，这个东西就是模拟，把$n$维的`object`拆成n维的点集，然后按照时间戳去模拟每个点的变化。之所以叫自动机，是因为每个点的变化是有规律，可以明确表示出来的，而且就算感性表示出来也行。
  * 比如说要建造最稳定沙堆的例子，你用物理规律去算水-沙混合物很难吧，但是你对每个水分子，对每个沙分子，感性的去描述他们之间的作用关系，比如，我如果一个水分子放在那里，那他会粘合旁边的沙子，你就可以直接线性的对两个分子进行操作，这样的操作累加到每个单位元胞上（水单位元胞，空气单位元胞，沙单位元胞）从而决定他们在下个时间点的位置。那些操作的系数可以试出来的，就比如我试了下系数组$(\lambda1,\lambda 2,\lambda3)$,结果发现他让沙堆不符合生活实际，我就调节三个变量，让沙堆符合生活实际。
  * 所以本质就是模拟。
  * 但这种模拟的策略是很好的，这解决了很大**一类问题**，而不是某个具体的问题，所以这是个**总体的策略**，**没有所谓具体的实现**，因为具体实现其实就是数组操作，这个变化很大的。