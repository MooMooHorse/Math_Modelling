# Genetic Algorithm

## 适用范围

* 在了解这个算法如何解决问题的时候，一定要知道他产生的后果是什么
* 我看很多论文就 乱用
  * 乱用的后果
    * **并不是正确性损失**，正确性是绝对可以保证的
    * 但是这个方法如果乱用，会**退化成直接随机**
    * 那是真的拼运气

在了解GA(genetic algorithm)之前一定要了解，为什么要用GA。

## 意义



### 先考虑暴力枚举

如果你想求一个函数的**最值**，那最基本的方法是枚举在定义域 $D$ 中自变量

### 产生的问题

如果你有N个点，产生一条路径，每个路径被$f(path)$ 映射到$\R$上的一个值

N小幅度增加会造成$D$大范围扩大，导致无法承受的算力



### 随机

可以想到，只随机产生一个集合$D_{sub} \sube D$, 然后$|D_{sub}|<<|D|$，我们只考虑$D_{sub}$中元素的值，这样就很大程度上减少了运算量，但不能保证正确。

什么时候**几乎**能保证正确呢？

当$f(D)$取值很平均，很单一的时候。

然后我们`随机很多次`，就能大概率保证正确性



### 改善

以上说的随机很多次，是独立的随机，那能不能把下一次随机基于上一次随机呢？

可以想到达尔文进化论，如果把自变量看成染色体上的基因，函数$f$看成表达后产生的性征，是不是可以模拟自然选择，第一次先随机，然后第二次只杂交第一次性征很优秀的一些定义域里的点，他们的后代就是第二次的自变量。

**更详细了解这一过程看**

https://blog.csdn.net/Hanpu_Liang/article/details/78172537?spm=1001.2014.3001.5501

### 局限性

如果X是钢琴师，Y是小提琴家，我们评价人评价他的能力，假设音乐家的能力严格小于工程师，并且基因大概率遗传，那X和Y的后代是不是极大概率达不到最优？而且是离最优小个十万八千里。但他在他们的行业里是最优。



这实际上对应的是，如果通过上一次随机得到这一次随机，那么你最终优化出来的一个点是`local extrema`



有没有发现，这个`GA` 算法其实效果等效于老白讲的爬山算法？

你到山顶自然选择，只能求`local extrema`



> 可以采用，**多次随机初始值**的方法解决这个问题



**局限性**会在`实质`中具体阐述

## 实质

就是爬山，如果山峰很少，那这样可以极大减少运算次数。

**但是如果山峰(local extrema)很多**，这个算法就会**退化成随机**，你找到最优解的概率几乎是0。

* 所以实质就是**随机**策略的一种**优化**。

## 实现

```matlab
function y = PlotModel(chrom)
x = chrom(1);
y = chrom(2);
z = chrom(3);
figure(2)
scatter3(x, y, z, 'ko')
hold on
[X, Y] = meshgrid(-10:0.1:10);
Z =sin(X)+cos(Y)+0.1*X+0.1*Y;
mesh(X, Y, Z)
y=1;

function [chrom_new, fitness_new] = ReplaceWorse(chrom, chrom_best, fitness)

max_num = max(fitness);
min_num = min(fitness);
limit = (max_num-min_num)*0.2+min_num;

replace_corr = fitness<limit;

replace_num = sum(replace_corr);
chrom(replace_corr, :) = ones(replace_num, 1)*chrom_best(1:end-1);
fitness(replace_corr) = ones(replace_num, 1)*chrom_best(end);
chrom_new = chrom;
fitness_new = fitness;

end

function chrom_new = MutChrom(chrom, mut, N, N_chrom, chrom_range, t, iter)
for i = 1:N
    for j = 1:N_chrom
        mut_rand = rand; %是否变异
        if mut_rand <=mut
            mut_pm = rand; %增加还是减少
            mut_num = rand*(1-t/iter)^2;
            if mut_pm<=0.5
                chrom(i, j)= chrom(i, j)*(1-mut_num);
            else
                chrom(i, j)= chrom(i, j)*(1+mut_num);
            end
            chrom(i, j) = IfOut(chrom(i, j), chrom_range(:, j)); %检验是否越界
        end
    end
end
chrom_new = chrom;

function chrom_new = AcrChrom(chrom, acr, N, N_chrom)
for i = 1:N
    acr_rand = rand;
    if acr_rand<acr %如果交叉
        acr_chrom = floor((N-1)*rand+1); %要交叉的染色体
        acr_node = floor((N_chrom-1)*rand+1); %要交叉的节点
        %交叉开始
        temp = chrom(i, acr_node);
        chrom(i, acr_node) = chrom(acr_chrom, acr_node); 
        chrom(acr_chrom, acr_node) = temp;
    end
end
chrom_new = chrom;

function c_new = IfOut(c, range)
if c<range(1) || c>range(2)
    if abs(c-range(1))<abs(c-range(2))
        c_new = range(1);
    else
        c_new = range(2);
    end
else
    c_new = c;
end

function fitness_ave = CalAveFitness(fitness)
[N ,~] = size(fitness);
fitness_ave = sum(fitness)/N;

function chrom_best = FindBest(chrom, fitness, N_chrom)
chrom_best = zeros(1, N_chrom+1);
[maxNum, maxCorr] = max(fitness);
chrom_best(1:N_chrom) =chrom(maxCorr, :);
chrom_best(end) = maxNum;

function fitness = CalFitness(chrom, N, N_chrom)
fitness = zeros(N, 1);
%开始计算适应度
for i = 1:N
    x = chrom(i, 1);
    y = chrom(i, 2);
    fitness(i) = sin(x)+cos(y)+0.1*x+0.1*y;
end

function chrom_new = Initialize(N, N_chrom, chrom_range)
chrom_new = rand(N, N_chrom);
for i = 1:N_chrom %每一列乘上范围
    chrom_new(:, i) = chrom_new(:, i)*(chrom_range(2, i)-chrom_range(1, i))+chrom_range(1, i);
end

clc, clear, close all

%%基础参数
N = 100;  %种群内个体数目
N_chrom = 2; %染色体节点数
iter = 2000; %迭代次数
mut = 0.2;  %突变概率
acr = 0.2; %交叉概率
best = 1;

chrom_range = [-10 -10;10 10];%每个节点的值的区间
chrom = zeros(N, N_chrom);%存放染色体的矩阵
fitness = zeros(N, 1);%存放染色体的适应度
fitness_ave = zeros(1, iter);%存放每一代的平均适应度
fitness_best = zeros(1, iter);%存放每一代的最优适应度
chrom_best = zeros(1, N_chrom+1);%存放当前代的最优染色体与适应度

%%初始化
chrom = Initialize(N, N_chrom, chrom_range); %初始化染色体
fitness = CalFitness(chrom, N, N_chrom); %计算适应度
chrom_best = FindBest(chrom, fitness, N_chrom); %寻找最优染色体
fitness_best(1) = chrom_best(end); %将当前最优存入矩阵当中
fitness_ave(1) = CalAveFitness(fitness); %将当前平均适应度存入矩阵当中

for t = 2:iter
    chrom = MutChrom(chrom, mut, N, N_chrom, chrom_range, t, iter); %变异
    chrom = AcrChrom(chrom, acr, N, N_chrom); %交叉
    fitness = CalFitness(chrom, N, N_chrom); %计算适应度
    chrom_best_temp = FindBest(chrom, fitness, N_chrom); %寻找最优染色体
    if chrom_best_temp(end)>chrom_best(end) %替换掉当前储存的最优
        chrom_best = chrom_best_temp;
    end
    %%替换掉最劣
    [chrom, fitness] = ReplaceWorse(chrom, chrom_best, fitness);
    fitness_best(t) = chrom_best(end); %将当前最优存入矩阵当中
    fitness_ave(t) = CalAveFitness(fitness); %将当前平均适应度存入矩阵当中
end

%%作图
figure(1)
plot(1:iter, fitness_ave, 'r', 1:iter, fitness_best, 'b')
grid on
legend('平均适应度', '最优适应度')
e = PlotModel(chrom_best)

%%输出结果
disp(['最优染色体为', num2str(chrom_best(1:end-1))])
disp(['最优适应度为', num2str(chrom_best(end))])
```

只需要更改`CalFitness`函数里面的$f(x)$定义即可解决所有该算法能解决的所有问题。

### 几个让你可能非常迷惑的代码里的点

* 如果看了原帖的话，不是说好，烂的个体突变概率大吗，怎么到代码里就是最开始突变概率大，iteration次数越来越多，突变概率越来越小，这合理吗？
  * 合理，首先烂的个体突变概率大是原作者意淫出来的，你想想高中课本上，突变概率都是一样的
  * 其次，为什么最开始突变概率大，不是说好突变概率一样吗？
    * 最开始的时候，地球上只有那些单细胞生物，然后无脊椎，然后blahblahblah，他们突变的概率确实一样，但是，要改清楚我们的目标，是进化的最终产物，也就是如果从进化论来说，我们的算法应该快速模拟出人这一物种。**等效的**，前几亿年的微小变异对我们来说一点用都没有，不妨直接将他们累加，或者说压缩，这样就可以看成，最开始变异率大（因为是很多小变异叠加出来的），最后变异率小
* 为什么这里染色体交叉这么奇怪，N个个体怎么产生N个后代？
  * 因为，他想模拟的是一个等效交配。
    * 什么叫等效交配？
    * 首先先看直接模拟交配为什么是不优的
      * 这样你每次都要$N\times N $ 交配，里面复杂度太高
    * 实际上$A$ $B$交配，无非就是子代取一部分$A$的基因一部分$B$的基因，然后随机组合。
    * **实质是基因库的不变性**
    * 所以等效交配就是在基因库不变的情况，随便乱分即可。所以可以N个个体一起生个小朋友，这也是允许的，因为这种情况就是等效两两互相交配，然后后代再两两互相交配，直至有个个体祖先是这N个个体。
* 这样看来，他的突变和交换都没错，而且基本上完整模拟了需要功能。
* 注意 matlab数组从 1开始编号
* 注意他chrom 数组1~size-1放的是基因，最后一个放的是适应度。
* 注意 数组<k 会得到一个与数组同size的逻辑数组 满足条件是 1 不满足条件是0 设这个逻辑数组为 $l$ ,原数组$A$
* $A(l)$会筛选出$l$中为1的数，比较智能吧