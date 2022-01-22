"""
    Use topsis to rank a set of data

    Link: https://zhuanlan.zhihu.com/p/37738503

"""





"""
    data direction

    See the link above for more detail
"""

def dataDirection_1(datas, offset=0):
	def normalization(data):
		return 1 / (data + offset)

	return list(map(normalization, datas))


def dataDirection_2(datas, x_min, x_max):
	def normalization(data):
		if data <= x_min or data >= x_max:
			return 0
		elif data > x_min and data < (x_min + x_max) / 2:
			return 2 * (data - x_min) / (x_max - x_min)
		elif data < x_max and data >= (x_min + x_max) / 2:
			return 2 * (x_max - data) / (x_max - x_min)

	return list(map(normalization, datas))


def dataDirection_3(datas, x_min, x_max, x_minimum, x_maximum):
	def normalization(data):
		if data >= x_min and data <= x_max:
			return 1
		elif data <= x_minimum or data >= x_maximum:
			return 0
		elif data > x_max and data < x_maximum:
			return 1 - (data - x_max) / (x_maximum - x_max)
		elif data < x_min and data > x_minimum:
			return 1 - (x_min - data) / (x_min - x_minimum)

	return list(map(normalization, datas))

"""
Topsis
"""


import pandas as pd
import numpy as np




def entropyWeight(data):
    """
        A way to calcuate weight
        see the link for more detail
    """

    data = np.array(data)
	# 归一化
	
    P = data / data.sum(axis=0)

	# 计算熵值
	
    E = np.nansum(-P * np.log(P) / np.log(len(data)), axis=0)

	# 计算权系数
	
    return (1 - E) / (1 - E).sum()



import numpy as np

RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51}
#RI 是随机一致性指标(查表得到)

def ahp(data):
    """
        以下“指标”指特征
        AHP确定特征之间相对权重
        输入data方阵
        data方阵定义: data[i,j] 表示指标 i 和指标 j 的相对重要程度，如 data[i,j]=1/3 表示指标 i 和指标 j 的相对重要程度为 1:3

        （一般都是查阅文献，xx 比 xx 重要再人为设定相对重要程度，一般以 1、3、5、7 界定重要程度）
    """

    data = np.array(data)
    m = len(data)

    # 计算特征向量
    weight = (data / data.sum(axis=0)).sum(axis=1) / m

    # 计算特征值
    Lambda = sum((weight * data).sum(axis=1) / (m * weight))

    # 判断一致性
    CI = (Lambda - m) / (m - 1)
    CR = CI / RI[m]

    if CR < 0.1:
        print(f'最大特征值:lambda = {Lambda}')
        print(f'特征向量:weight = {weight}')
        print(f'\nCI = {round(CI,2)}, RI = {RI[m]} \nCR = CI/RI = {round(CR,2)} < 0.1，通过一致性检验')
        return weight
    else:
        print(f'\nCI = {round(CI,2)}, RI = {RI[m]} \nCR = CI/RI = {round(CR,2)} >= 0.1，不满足一致性')



def topsis(data, weight=None):
	# 归一化
	data = data / np.sqrt((data ** 2).sum())

	# 最优最劣方案
	Z = pd.DataFrame([data.min(), data.max()], index=['Dminus', 'Dpositive'])

	# 距离
	weight = entropyWeight(data) if weight is None else np.array(weight)
	Result = data.copy()
	Result['Dpositive'] = np.sqrt(((data - Z.loc['Dpositive']) ** 2 * weight).sum(axis=1))
	Result['Dminus'] = np.sqrt(((data - Z.loc['Dminus']) ** 2 * weight).sum(axis=1))

	# 综合得分指数
	Result['C'] = Result['Dminus'] / (Result['Dminus'] + Result['Dpositive'])
	Result['Rank'] = Result.rank(ascending=False)['C']

	return Result, Z, weight


"""
    data 格式如下，将字典前替换成特征名字 字典后替换成数据值即可
"""

data = pd.DataFrame(
    {'人均专著': [0.1, 0.2, 0.4, 0.9, 1.2], '生师比': [5, 6, 7, 10, 2], '科研经费': [5000, 6000, 7000, 10000, 400],
     '逾期毕业率': [4.7, 5.6, 6.7, 2.3, 1.8]}, index=['院校' + i for i in list('ABCDE')])

"""
    预处理完毕
"""


"""
    数据同向化,按照题目和以上链接 选取三种方式之一
"""
data['生师比'] = dataDirection_3(data['生师比'], 5, 6, 2, 12)   # 师生比数据为区间型指标
data['逾期毕业率'] = 1 / data['逾期毕业率']   # 逾期毕业率为极小型指标

"""

    剩下的标准操作，唯一需要改的就是权重
    可以直接给权重，也可以通过AHP选权重，也可以用EW选，用后两者选的时候把weight=None  然后再函数内对应那行改下即可
"""
out = topsis(data, weight=[0.2, 0.3, 0.4, 0.1])    # 设置权系数 可用AHP 也可用EW

print(out)
