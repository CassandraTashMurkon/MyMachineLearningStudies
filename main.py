import pandas as pd  
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


#dataloader
df = quandl.get('WIKI/GOOGL')


#dataframe preperation
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] -df['Adj. Close'] ) / df['Adj. Close'] * 100.0
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open'] ) / df['Adj. Open'] * 100.0

#new dataframe
df = df[['Adj. Close','HL_PCT','PCT_Change','Adj. Volume']]



forecast_col = 'Adj. Close'
df.fillna(-9999, inplace=True)

forcast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(forcast_out)



#Features 
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forcast_out]
X_lately = X[-forcast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])
#Spit the data into test and training data the Float defines the percentage
X_train, X_test, y_train, y_test = model_selection.train_test_split(X , y , test_size=0.2)

# The Algorithm
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test,y_test)

#Prediction
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forcast_out)

#Creating graph
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
Next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(Next_unix)
    Next_unix += one_day
    df.loc[next_date] =[np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


