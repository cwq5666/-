import numpy as np
#为防止log(0)的出现，将0替换为1
def mylog(x):
    x = list(x)
    for i in range(len(x)):
        if x[i] == 0:
            x[i] = 1
    return np.log(x)




# 定义一个指标矩阵X
X = np.array([[9,0,0,0],[8,3,0.9,0.5],[6,7,0.2,1]])

#标准化矩阵
Z = X/np.sqrt(np.sum(X**2,axis = 0))
print("标准化后的矩阵",Z)

n,m = Z.shape#获取行数和列数
D = np.zeros(m)

#计算信息效用值
for i in range(m):
    x = Z[ :,i]#获取z的第i列
    p = x/np.sum(x) #归一化
    #使用自己的函数计算信息熵
    D[i] =1 + np.sum(p*mylog(p))/np.log(n)
#计算权重
W = D/np.sum(D)
print("权重为：",W)