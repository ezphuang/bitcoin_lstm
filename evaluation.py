import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from empyrical import max_drawdown as mdd1

revenue = pd.read_csv('E_vals.csv').iloc[:, 1]
a = revenue.tolist()
b=revenue/1000


def mdd(array):
    drawdowns = []
    max_so_far = array[0]
    for i in range(len(array)):
        if array[i] > max_so_far:
            drawdown = 0
            drawdowns.append(drawdown)
            max_so_far = array[i]
        else:
            drawdown = max_so_far - array[i]
            drawdowns.append(drawdown)
    return max(drawdowns)


def pp(list):
    positive=[]
    negative=[]
    a=len(list)
    for i in range(1,a):
        rate=list[i]/list[i-1]-1
        if rate>=0:
            positive.append(rate)
        else:
            negative.append(rate)
    percen=len(positive)/(a-1)
    return percen

def pf(list):
    positive = []
    negative = []
    a = len(list)
    for i in range(1, a):
        gross = list[i]-list[i - 1]
        if gross >= 0:
            positive.append(gross)
        else:
            negative.append(gross)
    pf = sum(positive) / abs(sum(negative))
    return pf

def npp(list):
    final=list[-1]
    net=final/1000*100
    return net


def mdd2(return_list):
    # 1. find all of the peak of cumlative return
    maxcum = np.zeros(len(return_list))
    b = return_list[0]
    for i in range(0, len((return_list))):
        if (return_list[i] > b):
            b = return_list[i]
        maxcum[i] = b

    # 2. then find the max drawndown point
    i = np.argmax((maxcum - return_list) / maxcum)
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])

    # 3. return the maxdrawndown
    return (return_list[j] - return_list[i]) / return_list[j]

print(a)
print(mdd2(a))
print(pf(a))
print(pp(a))
print(npp(a))














