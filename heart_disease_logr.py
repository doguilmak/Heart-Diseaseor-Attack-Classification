# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 22:27:17 2021

@author: doguilmak

dataset: https://www.kaggle.com/alexteboul/heart-disease-health-indicators-dataset

dataset info: https://www.kaggle.com/alexteboul/heart-disease-health-indicators-dataset-notebook

"""
#%%
# 1. Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

#%%
# 2. Data Preprocessing

# 2.1. Uploading data
start = time.time()
df = pd.read_csv('heart_disease_health_indicators_BRFSS2015.csv')
print(list(df.columns))
print(df.head())
print(df.describe().T)

pd.crosstab(df.Sex,df.HeartDiseaseorAttack).plot(kind='bar')
plt.title('Heart Disease or Attack Frequency for Gender')
plt.xlabel('Gender')
plt.ylabel('Frequency of Hearth Disease Attack')
plt.savefig('diseaseorAttack')
plt.show()

# 2.3. Dropping id Column(id)
#df.drop('Education', axis = 1, inplace = True)
#df.drop('Income', axis = 1, inplace = True)
print("Duplicated: {}".format(df.duplicated().sum()))

# 2.4. Looking for Duplicated Values
dp = df[df.duplicated(keep=False)]
dp.head(2)
df.drop_duplicates(inplace= True)
print("Duplicated: {}".format(df.duplicated().sum()))

# 2.5. Determination of Dependent and Independent Variables
y = df["HeartDiseaseorAttack"]
X = df.drop("HeartDiseaseorAttack", axis = 1)

# 2.6. Splitting test and train 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# 2.7. Scaling datas
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test) # Apply the trained

#%%
# Logistic Regression

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(C=3, tol=0.0001)
logr.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logr.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logr.predict_proba(X_test)[:,1])
plt.figure(figsize=(12, 12))
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.3f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Logistic Regression')
plt.legend(loc='upper left')
plt.savefig('Log_ROC_curve')
plt.show()

y_pred = logr.predict(X_test)
print('Accuracy of logistic regression classifier: {:.3f}'.format(logr.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#%%
# K-Fold Cross Validation

from sklearn.model_selection import cross_val_score

success = cross_val_score(estimator = logr, 
                          X=X_train, 
                          y=y_train, 
                          cv = 4)

print("\nK-Fold Cross Validation:")
print("Success Mean:\n", success.mean())
print("Success Standard Deviation:\n", success.std())

# Grid Search
from sklearn.model_selection import GridSearchCV
p = [{'tol':[1e-4,1e-3,1e-2], 'C':[1,2,3]},
     {'tol':[1e-4,1e-3,1e-2], 'C':[1,2,3]},
     {'tol':[1e-4,1e-3,1e-2], 'C':[1,2,3]},
     {'tol':[1e-4,1e-3,1e-2], 'C':[1,2,3]}]


gs = GridSearchCV(estimator= logr,
                  param_grid=p,
                  scoring='accuracy',
                  cv=5,
                  n_jobs=-1)

grid_search = gs.fit(X_train, y_train)
best_result = grid_search.best_score_
best_parameters = grid_search.best_params_
print("\nGrid Search:")
print("Best result:\n", best_result)
print("Best parameters:\n", best_parameters)

end = time.time()
cal_time = end - start
print("\nProcess took {} seconds.".format(cal_time))