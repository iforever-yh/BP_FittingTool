import math
import numpy as np

#fun1 生成sin训练集
def gensin(min=0, max=1, step=0.5):
    x = []
    y = []
    for i in np.arange(min, max, step):
        x.append([i])
        y.append([math.sin(i)])

    return x, y

#fun2 生成cos训练集
def gencos(min=0, max=1, step=0.5):
    x = []
    y = []
    for i in np.arange(min, max, step):
        x.append([i])
        y.append([math.cos(i)])

    return x, y

#fun3 生成power训练集
def genpower(min=0, max=1, step=0.5, power=1):
    x = []
    y = []
    for i in np.arange(min, max, step):
        x.append([i])
        y.append([math.pow(i, power)])
        
    return x, y