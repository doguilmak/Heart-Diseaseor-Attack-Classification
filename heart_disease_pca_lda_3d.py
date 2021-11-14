# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:31:20 2021

@author: doguilmak

dataset: https://www.kaggle.com/alexteboul/heart-disease-health-indicators-dataset

dataset info: https://www.kaggle.com/alexteboul/heart-disease-health-indicators-dataset-notebook

"""

#%%
#  1. Libraries

import pandas as pd
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
X_test = sc.transform(x_test)

#%%
# 3. PCA

from sklearn.decomposition import PCA
pca = PCA(n_components = 3)  # 3 dimensional

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

principalDf = pd.DataFrame(data = X_train2,
              columns = ['principal component 1', 'principal component 2', 
                         'principal component 3'])
finalDf = pd.concat([principalDf, df[['HeartDiseaseorAttack']]], axis = 1)

fig = plt.figure(figsize = (12, 12))
ax = Axes3D(fig)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 Component PCA', fontsize = 20)
targets = [0, 1, 2]
colors = ['r', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['HeartDiseaseorAttack'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , finalDf.loc[indicesToKeep, 'principal component 3']
               , c = color
               , s=2)
ax.legend(['Diagnose: Healthy', 'Diagnose: Heart Diseaseor Attack'])
#plt.savefig('Plots/3_Component_PCA')

"""
# Plotting for GIF
import numpy as np
for ii in np.arange(0, 360, 1):
    ax.view_init(elev=32, azim=ii)
    fig.savefig('Plots/3D_plot_figures/gif_image%d.png' % ii)

ax.grid()
"""

# 3.2. LR Transform Before PCA
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0, C=1, tol=0.0001)
classifier.fit(X_train, y_train)

# 3.3. LR After PCA Transform
classifier2 = LogisticRegression(random_state=0, C=1, tol=0.0001)
classifier2.fit(X_train2, y_train)

# 3.4. Predictions
y_pred = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# 3.5. Actual / Without PCA 
print('Actual / Without PCA')
cm1 = confusion_matrix(y_test, y_pred)
print(cm1)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}\n")

# 3.6. Actual / Result after PCA
print("Actual / With PCA")
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)
print(f"Accuracy score: {accuracy_score(y_test, y_pred2)}\n")

# 3.7. After PCA / Before PCA
print('Before PCA and After PCA')
cm3 = confusion_matrix(y_pred, y_pred2)
print(cm3)
print(f"Accuracy score: {accuracy_score(y_pred, y_pred2)}\n")

end = time.time()
cal_time = end - start
print("\nProcess took {} seconds.".format(cal_time))
