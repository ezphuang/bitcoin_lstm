import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

factorbook = pd.read_csv('factors5_no.csv')
print(factorbook)
pricebook = pd.read_csv('prices.csv')


def iftrade(day): #False 黄金交易日
    gold_price = pricebook.loc[day, 'gold']
    g = np.isnan(gold_price)
    return g

def get_factors(type,day):
    allfactors = factorbook.iloc[day, 1:].values
    if type=="gold":
        factorvector = allfactors[5:]
    else:
        factorvector = allfactors[:5]
    return factorvector

def get_price(type,day):
    price = pricebook.loc[day, type]
    return price

def prices(day):
    prices = pricebook.iloc[day, 1:].values
    return prices

def values(holding,day): #holding[cash,gold,bit]
    if iftrade(day)==False:
       price = prices(day).tolist()
       price.insert(0, 1)
       print(price)
       values = np.multiply(np.array(holding),np.array(price))
       values = sum(values.tolist())
    else:
        price = prices(day).tolist()
        price_bit=price[1]
        while np.isnan(price[0]) == True:
            day = day-1
            price = prices(day).tolist()
        price_gold=price[0]
        price=[1,price_gold,price_bit]
        values = np.multiply(np.array(holding), np.array(price))
        values = sum(values.tolist())

    return values

def signal(type,day):
    if type=="gold":
        weight=weight_gold
    else:
        weight=weight_bit
    factors = get_factors(type,day)
    s = np.multiply(np.array(weight), np.array(factors))
    D = sum(s.tolist())
    return D

buy_co = [0.2,0.8]
sell_co = [0.1,0.01]
holding=[1000,0,0]

weight_gold=[0.58,0.35,0.01,0.01,0.05]
weight_bit=[0.15,0.05,0.45,0.15,0.2]
#weight_gold=[0.67,0.33,0,1,0]
#weight_bit=[0.67,0.33,0,1,0]
#tradingfee=[0.01,0.02]
tradingfee=[0,0]
cash=[]
gold=[]
bit=[]
val=[]

for i in range(151,1826):
    cash.append(holding[0])
    gold.append(holding[1])
    bit.append(holding[2])
    print(i)
    j=0
    newholding=holding
    value=values(holding,i)
    if iftrade(i)==False:
        for g in ["gold","bit"]:
            j=j+1
            D = signal(g,i)
            price = get_price(g,i)
            print(D)
            if D>=0:
                buyin = (holding[0] * buy_co[j-1] * D) / (price *(1+tradingfee[j-1]) ) #买入量
                newholding[0] = newholding[0] - (holding[0] * buy_co[j-1] * D)
                newholding[j] = newholding[j] + buyin
            else:
                sellout = holding[j] * sell_co[j-1] *D #卖出量
                newholding[0] = newholding[0] - (sellout*price*(1-tradingfee[j-1]))
                newholding[j] = newholding[j] + sellout
    else:
        j = 2
        g = "bit"
        D = signal(g, i)
        print(D)
        price = get_price(g, i)
        if D >= 0:
            buyin = (holding[0] * buy_co[j-1] * D) / (price *(1+tradingfee[j-1]) )  # 买入量
            newholding[0] = newholding[0] - (holding[0] * buy_co[j-1] * D)
            newholding[j] = newholding[j] + buyin
        else:
            sellout = holding[j] * sell_co[j-1] * D  # 卖出量
            newholding[0] = newholding[0] - (sellout * price * (1 - tradingfee[j - 1]))
            newholding[j] = newholding[j] + sellout
    print(value)
    val.append(value)
    print(holding)
    holding=newholding

holdings = pd.DataFrame(cash, columns=['cash'])
holdings = pd.concat([holdings, pd.DataFrame(gold,columns=['gold'])],axis=1)
holdings = pd.concat([holdings, pd.DataFrame(bit,columns=['bit'])],axis=1)
vals = pd.DataFrame(val, columns=['value'])
print(holdings)
print(vals)

#holdings.to_csv("ran_0.05_holding.csv")
vals.to_csv("no_cost_adj.csv")


















