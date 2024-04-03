# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:13:33 2024

@author: berre
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(path)
df.head()
print(df.dtypes)
#correlation between the following columns: bore, stroke, compression-ratio, and horsepower.
df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()
#"regplot" which plots the scatterplot plus the fitted regression line for the data
# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)
df[['engine-size','price']].corr()
sns.regplot(x="highway-mpg", y="price", data=df)
df[['highway-mpg', 'price']].corr()
sns.regplot(x="peak-rpm", y="price", data=df)
#Value counts is a good way of understanding how many units of each characteristic/variable we have
#"value_counts" only works on pandas series, not pandas dataframes

drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)
# "groupby" method groups data by different categories. The data is grouped based on one or several variables, and analysis is performed on the individual groups.
df['drive-wheels'].unique()
#on average, which type of drive wheel is most valuable, we can group "drive-wheels" and then average them
df_group_one = df[['drive-wheels','body-style','price']]
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
df['price'].unique()
