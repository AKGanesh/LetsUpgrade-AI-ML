# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 18:22:10 2020

@author: Ganesh
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


dataset = pd.read_csv("train.csv");

print(dataset.columns)

le = preprocessing.LabelEncoder()
le.fit(dataset["Sex"])
print(le.classes_)

dataset["Sex"] = le.transform(dataset["Sex"])
dataset["Embarked"] = le.fit_transform(dataset["Embarked"])

datasetOriginal = dataset

def print_scores(datasetOriginal, dv, idv_list):
    y = datasetOriginal[dv]
    X = datasetOriginal.drop(idv_list, axis=1)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    clf = BernoulliNB()
    y_pred = clf.fit(X_train, y_train).predict(X_test)

    print(accuracy_score(y_test, y_pred, normalize=True))
    print(confusion_matrix(y_test, y_pred))
# =============================================================================
# y = dataset["Survived"]
# X = dataset.drop(["Survived","PassengerId", "Name", "Ticket", "Cabin"], axis=1)
# 
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
# 
# clf = BernoulliNB()
# 
# y_pred = clf.fit(X_train, y_train).predict(X_test)
# 
# print(accuracy_score(y_test, y_pred, normalize=True))
# print(confusion_matrix(y_test, y_pred))
# =============================================================================

print_scores(datasetOriginal, "Survived", ["Survived","PassengerId", "Name", "Ticket", "Cabin"])
print_scores(datasetOriginal, "Pclass", ["Pclass","PassengerId", "Name", "Ticket", "Cabin"])
print_scores(datasetOriginal, "Sex", ["Sex","PassengerId", "Name", "Ticket", "Cabin"])
print_scores(datasetOriginal, "SibSp", ["SibSp","PassengerId", "Name", "Ticket", "Cabin"])
print_scores(datasetOriginal, "Parch", ["Parch","PassengerId", "Name", "Ticket", "Cabin"])

# =============================================================================
# 0.7715355805243446
# [[131  26]
#  [ 35  75]]
# 
# 0.5917602996254682
# [[ 30   8  32]
#  [  9  10  30]
#  [ 24   6 118]]
# 
# 0.7453183520599251
# [[ 49  49]
#  [ 19 150]]
# 
# 0.6891385767790262
# [[162  20   0   0   0   0   0]
#  [ 43  22   0   0   0   0   0]
#  [  6   2   0   0   0   0   0]
#  [  5   2   0   0   0   0   0]
#  [  2   0   0   0   0   0   0]
#  [  1   0   0   0   0   0   0]
#  [  2   0   0   0   0   0   0]]
# 
# 0.7153558052434457
# [[182  16   0   0   0]
#  [ 31   9   0   0   0]
#  [ 24   3   0   0   0]
#  [  1   0   0   0   0]
#  [  1   0   0   0   0]]
# =============================================================================
