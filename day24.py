# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 19:24:56 2020

@author: Ganesh
"""
"""
Project 1: Build Decision Tree(DV-"Survived",IDV-"Age,Gender and Fare") and Prediction

"""

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn import tree

titanic_train = pd.read_csv("train.csv")

new_age_var = np.where(titanic_train["Age"].isnull(), 28,titanic_train["Age"])
titanic_train["Age"] = new_age_var

label_encoder = preprocessing.LabelEncoder()

encoded_sex = label_encoder.fit_transform(titanic_train["Sex"])

tree_model = tree.DecisionTreeClassifier(max_depth=6)

predictors = pd.DataFrame([encoded_sex, titanic_train["Age"], titanic_train["Fare"]]).T

tree_model.fit(X=predictors, y=titanic_train["Survived"])

with open("Dtree1.dot", 'w') as f:
    f = tree.export_graphviz(tree_model, feature_names=["Sex","Age","Fare"], out_file=f);
    
print(tree_model.score(X=predictors,y=titanic_train["Survived"]))

titanic_test = pd.read_csv("test.csv")


new_age_var = np.where(titanic_train["Age"].isnull(), 28,titanic_train["Age"])
titanic_train["Age"] = new_age_var

label_encoder = preprocessing.LabelEncoder()

encoded_sex = label_encoder.fit_transform(titanic_train["Sex"])

test_features = pd.DataFrame([encoded_sex, titanic_test["Age"], titanic_test["Fare"]]).T

test_preds = tree_model.predict(X=test_features)

Predicted_Output = pd.DataFrame({"PassengerId":titanic_test["PassengerId"], "Survived":test_preds})

Predicted_Output.to_csv("Output.csv", index=False)


# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 20:48:24 2020

@author: Ganesh
"""

"""
Project 3: Build Decision Tree for Bank Loan Modelling
DV - "Personal Loan"
IDV - Output of RF Algorithm 

"""

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

bankdata = pd.read_excel("Bank_Personal_Loan_Modelling.xlsx","Data")
print(bankdata.columns)

label_encoder = preprocessing.LabelEncoder()


rf_model = RandomForestClassifier(n_estimators=1000, max_features=2, oob_score=True)

features = ['Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg',
       'Education', 'Mortgage', 'Securities Account',
       'CD Account', 'Online', 'CreditCard']

rf_model.fit(X=bankdata[features], y=bankdata['Personal Loan'])

print("OOB Accuracy:", rf_model.oob_score_)


for features, imp in zip(features, rf_model.feature_importances_):
    print(features, imp)
    
"""
OOB Accuracy: 0.9866
Age 0.04190316873390942
Experience 0.04085342032147368
Income 0.3302798787224294 **
ZIP Code 0.044569595859197696
Family 0.09261282466450306
CCAvg 0.17920434346439648 **
Education 0.15064171996655978 **
Mortgage 0.04365153644305667
Securities Account 0.00563781182229569
CD Account 0.052978394445300024
Online 0.008120899299835569
CreditCard 0.00954640625704229
"""

tree_model = tree.DecisionTreeClassifier(max_depth=4)

predictors = pd.DataFrame([bankdata["Education"],bankdata["Income"], bankdata["CCAvg"]]).T

tree_model.fit(X=predictors, y=bankdata["Personal Loan"])

print(tree_model.score(X=predictors,y=bankdata["Personal Loan"]))

with open("DtreeBank1.dot", 'w') as f:
    f = tree.export_graphviz(tree_model, feature_names=["Education","Income","CCAvg"], out_file=f);
    
""" 97 percent accurate with 2 IDVs, 92% if we take only 2 IDVs """

'''
Project 2: Build Decision Tree for Attrition Rate Analysis
DV - "Attrition"
IDV - Output of RF Algorithm
'''

attritiondata = pd.read_csv("general_data.csv")
print(attritiondata.columns)

label_encoder = preprocessing.LabelEncoder()
attritiondata['BusinessTravel'] =  label_encoder.fit_transform(attritiondata['BusinessTravel'])
attritiondata['Department'] =  label_encoder.fit_transform(attritiondata['Department'])
attritiondata['EducationField'] =  label_encoder.fit_transform(attritiondata['EducationField'])
attritiondata['Gender'] =  label_encoder.fit_transform(attritiondata['Gender'])
attritiondata['JobRole'] =  label_encoder.fit_transform(attritiondata['JobRole'])
attritiondata['MaritalStatus'] =  label_encoder.fit_transform(attritiondata['MaritalStatus'])
attritiondata['Over18'] =  label_encoder.fit_transform(attritiondata['Over18'])
attritiondata['Attrition'] =  label_encoder.fit_transform(attritiondata['Attrition'])

workingyrsmean = attritiondata['TotalWorkingYears'].mean()
companiesworkedmean = attritiondata['NumCompaniesWorked'].mean()


attritiondata['TotalWorkingYears'] = np.where(attritiondata['TotalWorkingYears'].isnull(), workingyrsmean, attritiondata['TotalWorkingYears'])
attritiondata['NumCompaniesWorked'] = np.where(attritiondata['NumCompaniesWorked'].isnull(),companiesworkedmean , attritiondata['NumCompaniesWorked'])


predictors = ['Age', 'BusinessTravel', 'Department', 'DistanceFromHome',
       'Education', 'EducationField', 'EmployeeCount', 'EmployeeID', 'Gender',
       'JobLevel', 'JobRole', 'MaritalStatus', 'MonthlyIncome',
       'NumCompaniesWorked', 'Over18', 'PercentSalaryHike', 'StandardHours',
       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

rf_model = RandomForestClassifier(n_estimators=4000, max_features=4, oob_score=True)

rf_model.fit(X=attritiondata[predictors], y=attritiondata['Attrition'])

print(rf_model.oob_score_)


for predictors, imp in zip(predictors, rf_model.feature_importances_):
    print(predictors, imp)

"""
Age 0.098337034061189**
BusinessTravel 0.02700237694117472
Department 0.024524430048534235
DistanceFromHome 0.06850808005334807*
Education 0.03739347125991716
EducationField 0.03886077497540824
EmployeeCount 0.0
EmployeeID 0.02943605613993381
Gender 0.015766061030027544
JobLevel 0.03453480633533145
JobRole 0.05294976999990188*
MaritalStatus 0.038726409118927066
MonthlyIncome 0.10114111262815699**
NumCompaniesWorked 0.05475088503184041*
Over18 0.0
PercentSalaryHike 0.06337239572361104*
StandardHours 0.0
StockOptionLevel 0.03192442026477167
TotalWorkingYears 0.0852699402721983**
TrainingTimesLastYear 0.042963838470526934
YearsAtCompany 0.06405341791900468
YearsSinceLastPromotion 0.041395010883494815
YearsWithCurrManager 0.049089708842701696
"""

tree_model = tree.DecisionTreeClassifier(max_depth=6)

predictors = pd.DataFrame([attritiondata["MonthlyIncome"],attritiondata["Age"], attritiondata["TotalWorkingYears"]]).T

tree_model.fit(X=predictors, y=attritiondata["Attrition"])

print(tree_model.score(X=predictors,y=attritiondata["Attrition"]))

with open("DtreeHR1.dot", 'w') as f:
    f = tree.export_graphviz(tree_model, feature_names=["MonthlyIncome","Age","TotalWorkingYears"], out_file=f);

""" Got the model accuracy/score as 86% """