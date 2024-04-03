# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:59:40 2024

@author: berre
"""

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score

path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
churn_df = pd.read_csv(path)


churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])


#normalize the data set

X1= preprocessing.StandardScaler().fit(X).transform(X)


#training
X_train, X_test, y_train, y_test = train_test_split( X1, y, test_size=0.2, random_state=4)
X_train0, X_test0, y_train0, y_test0 = train_test_split( X, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

print ('Train set:', X_train0.shape,  y_train0.shape)
print ('Test set:', X_test0.shape,  y_test0.shape)

#modeling-logistic regression
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

LR0 = LogisticRegression(C=0.01, solver='liblinear').fit(X_train0,y_train0)# not normalized

#prediction
yhat = LR.predict(X_test)
yhat0 = LR0.predict(X_test0) 
yhat1 = LR0.predict(X_test)
yhat2= LR.predict(X_test0)

yhat_prob = LR.predict_proba(X_test)
yhat_prob0 = LR0.predict_proba(X_test0)
yhat_prob1= LR0.predict_proba(X_test)
yhat_prob2= LR.predict_proba(X_test0)

#evaluation

j=jaccard_score(y_test, yhat,pos_label=0)
j0=jaccard_score(y_test0, yhat0,pos_label=0)
j1=jaccard_score(y_test, yhat1,pos_label=0)
j2=jaccard_score(y_test0, yhat2,pos_label=0)

######################################################################
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))


cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
cnf_matrix0 = confusion_matrix(y_test0, yhat0, labels=[1,0])
cnf_matrix1 = confusion_matrix(y_test, yhat1, labels=[1,0])
cnf_matrix2 = confusion_matrix(y_test0, yhat2, labels=[1,0])

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(1)
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

plt.figure(2)
plot_confusion_matrix(cnf_matrix0, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

plt.figure(3)
plot_confusion_matrix(cnf_matrix1, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

plt.figure(4)
plot_confusion_matrix(cnf_matrix2, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

print (classification_report(y_test, yhat))
print (classification_report(y_test0, yhat0))
print (classification_report(y_test, yhat1))
print (classification_report(y_test0, yhat2))


from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)


LR2 = LogisticRegression(C=0.01, solver='sag').fit(X_train,y_train)
yhat_prob2 = LR2.predict_proba(X_test)
print ("LogLoss: : %.2f" % log_loss(y_test, yhat_prob2))

