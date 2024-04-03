# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:21:27 2024

@author: berre
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"
df = pd.read_csv(filepath, header=0)
lm=LinearRegression()
X=df[['highway-mpg']]
Y=df[['price']]
lm.fit(X,Y)
Yhat=lm.predict(X)
#lm.intercept_
#lm.coef_
#Z=df[['horsepower','curb-weight','engine-size','highway-mpg']]
#lm.fit(Z, df['price'])
Yhat2=lm.predict(X)
#regression plot

sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
sns.residplot(df['highway-mpg'],df['price'])
#sns.residplot(x='highway-mpg',y='price',data=df)

#distribution plot
#ax1=sns.distplot(df['price'],hist=False,color="r",label="Actual Value",kde=True)
#sns.distplot(Yhat,hist=False,color="b",label="Fitted Values",ax=ax1,kde=True)
x=df["highway-mpg"]
y=df["price"]
#polynomial regression
f=np.polyfit(x,y,3)
p=np.poly1d(f)
print(p) # -1.557 x^3 + 204.8 x^2 - 8965 x + 1.379e+05

#pr=PolynomialFeatures(degree=2,include_bias=False)
#x_polly=pr.fit_transform(x[['horsepower','curb-weight']])
#SCALE=StandardScaler()
#SCALE.fit(x_data[['horsepower','highway-mpg']])
#x_scale=SCALE.transform(x_data[['horsepower','curb-weight']])

#Input=[('scale',StandardScaler()),('polynomial',PolynomialFeatures(degree=2)),('model',LinearRegression())]
#Pipe=Pipeline(Input)
#Pipe.fit(df[['horsepower','curb-weight','engine-size','highway-mpg']],y)
#yhat=Pipe.predict(X[['horsepower','curb-weight','engine-size','highway-mpg']])
lm.fit(df['highway-mpg'],df['price'])
lm.predict(np.array(30.0).reshape(-1,1))
new_input=np.arange(1,101,1).reshape(-1,1)