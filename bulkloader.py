import pandas as pd
import sys, datetime,glob
import numpy as np


workDir = 'C:\\Users\\Arnold\\OneDrive\\corp_manager\\Accounting\\MarketData\\Data\\'

myData = glob.glob(workDir+'*.csv')

df = pd.DataFrame()
for i in range(len(myData)):
    df2 =  pd.read_csv(myData[i], header=None)
    df = df.append(df2, ignore_index=True)
   # print(df.head(10))



df[2] = df[2].astype(str).str[:-4]
df[4] = df[4].astype(str).str[:-4]


sort = df.sort_values(by=[4])
buys =df[4].str.split('-').str[0]

#print (sort.head(10))
print (buys)
#df23 = df((df[4].astype(float))>0)
#print(df.head(5))
#print(df23)