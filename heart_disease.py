# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 20:08:34 2021

@author: doguilmak

dataset: https://www.kaggle.com/alexteboul/heart-disease-health-indicators-dataset

dataset info: https://www.kaggle.com/alexteboul/heart-disease-health-indicators-dataset-notebook

"""
#%%
# 1. Importing Libraries

import pandas as pd
import seaborn as sns
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import time
import warnings
warnings.filterwarnings('ignore')

#%%
# 2. Data Preprocessing

# 2.1. Uploading data
start = time.time()
df = pd.read_csv('heart_disease_health_indicators_BRFSS2015.csv')
print(df.head())
print(df.describe().T)

# 2.2. Plot Gender Pie Chart
explode = (0, 0.05)
fig = plt.figure(figsize = (12, 12), facecolor='w')
out_df=pd.DataFrame(df.groupby('Sex')['Sex'].count())

patches, texts, autotexts = plt.pie(out_df['Sex'], autopct='%1.1f%%',
                                    textprops={'color': "w"},
                                    explode=explode,
                                    startangle=90, shadow=True)

for patch in patches:
    patch.set_path_effects({path_effects.Stroke(linewidth=2.5,
                                                foreground='w')})

plt.legend(labels=['Female','Male'], bbox_to_anchor=(1., .95), title="Gender")
plt.savefig('gender_pie')
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
# 3 Artificial Neural Networks

# 3.1 Importing Libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout


# 3.1 Loading Created Model
#classifier = load_model('model.h5')

# 3.2 Checking the Architecture of the Model
#classifier.summary()

"""
# 3.2. Creating layers

Activations link: https://keras.io/api/layers/activations/
activation="sigmoid"
activation="relu"
activation="softmax"
softplus="softplus"
"""

classifier = Sequential()
# Input Layer(neurons) + Output Layer / 2 ----- (14 + 2) / 2
# Creating First Hidden Layer:
classifier.add(Dense(16, init="uniform", activation="relu", input_dim=21))
# Creating Second Hidden Layer:
classifier.add(Dense(16, init="uniform", activation="relu"))

classifier.add(Dense(256))
classifier.add(Dropout(0.25))

# Creating Output Layer:
classifier.add(Dense(1, init="uniform", activation="sigmoid"))

# 3.3. Train the Data
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])  # Compilation is done. It uses binary_crossentropy as the data is a binary result.
# Number of Epochs = 5
classifier_history = classifier.fit(X, y, epochs=16, batch_size=32, validation_split=0.13)  # Learn from Independent Variables

# 3.4. Predict
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# 3.5. Plot accuracy and val_accuracy
print(classifier_history.history.keys())
#classifier.summary()
#classifier.save('model.h5')

plt.figure(figsize=(12, 12))
sns.set_style('whitegrid')
plt.plot(classifier_history.history['accuracy'])
plt.plot(classifier_history.history['val_accuracy'])
plt.title('ANN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
plt.savefig('val_acc')
plt.show()

"""
from keras.utils import plot_model
plot_model(classifier, "binary_input_and_output_model.png", show_shapes=True)
"""

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# 3.6. Mean of Validation Accuracy
import statistics
mean_val_accuracy = statistics.mean(classifier_history.history['accuracy'])
print(f'\nMean of classifier accuracy: {mean_val_accuracy}')    

# 3.7. Accuracy Score
from sklearn.metrics import accuracy_score
print(f"\nAccuracy score: {accuracy_score(y_test, y_pred)}")

# 3.8. Accuracy Score
print('\nANN Prediction')
predict = X.iloc[0:1, ]
print(f'\nModel predicted as {classifier.predict_classes(predict)}.')

#%%
# 4 XGBoost

# 4.1 Importing Libraries
from xgboost import XGBClassifier

model=XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 4.2. Building Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_pred, y_test)  # Comparing results
print("\nConfusion Matrix(XGBoost):\n", cm2)

# 4.4. Accuracy Score
from sklearn.metrics import accuracy_score
print(f"\nAccuracy score(XGBoost): {accuracy_score(y_test, y_pred)}")

# 4.5. Prediction
print('\nXGBoost Prediction')
predict_model_XGBoost = X.iloc[0:1, ]
print(f'Model predicted class as {classifier.predict(predict_model_XGBoost)}.')

#%%
# 5. Visualize ANN
"""
from ann_visualizer.visualize import ann_viz

try:
    ann_viz(classifier, view=True, title="", filename="ann")
except:
    print("PDF saved.")
"""   
#%%

end = time.time()
cal_time = end - start
print("\nProcess took {} seconds.".format(cal_time))
