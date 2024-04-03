# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:27:00 2024

@author: berre
"""

import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
my_data = pd.read_csv(path,delimiter=",")


#pre processing
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

#converting sex and bp to numerical values using LabelEncoder
le_sex = preprocessing.LabelEncoder()

le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

y = my_data["Drug"]


#setting up the decision tree
#splitting the data
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))

print('Shape of X test set {}'.format(X_testset.shape),'&',' Size of Y tesy set {}'.format(y_testset.shape))

#modeling
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

drugTree.fit(X_trainset,y_trainset)
predTree = drugTree.predict(X_testset)

print (predTree [0:5])
print (y_testset [0:5])


print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
tree.plot_tree(drugTree)
plt.show()