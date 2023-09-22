# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 19:29:03 2023

@author: pujar
"""

# libraries imported 
'''import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy

# ...

# Building and Training Deep Training Model

from sklearn.model_selection import train_test_split # Data splitting 
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, recall_score # Evaluation Metrics
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier

# Training, predicting and evaluating baseline

# Splitting dataset into independent variables (X) and target variable (y)
X = instagram_df_train.drop('fake', axis = 1)
y = instagram_df_train['fake']

# Creating training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Initializing mode
rf = RandomForestClassifier(random_state = 42)
rf.fit(X_train, y_train) # Fitting to training data 

y_pred = rf.predict(X_val) # Predicting on validation set
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds': thresholds})

baseline_score = roc_auc_score(y_val, y_pred)
print('\n')
print('AUC-ROC Baseline: ', baseline_score.round(2))
print('\n')

sns.set_style('darkgrid')
sns.lineplot(x='FPR', y='TPR', data=roc_df, label=f'RandomForest Classifier(AUC-ROC = {baseline_score.round(2)})')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guessing')
plt.title('AUC-ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
print('\n')
print('y_val value counts')
print(y_val.value_counts())
print('\n')
print('predicted value counts')
print(np.unique(y_pred, return_counts=True))

print(classification_report(y_pred, y_val))

plt.figure(figsize=(10, 10))
cm=confusion_matrix(y_pred, y_val)
sns.heatmap(cm, annot=True)
plt.show()

# Save the trained model in HDF5 format
rf.save('your_model.h5')'''

# Import necessary libraries and load the training data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

# Load the training dataset
instagram_df_train = pd.read_csv('train.csv')

# Preprocessing: Drop unnecessary columns and scale the data
X = instagram_df_train.drop(columns=['fake'])
y = instagram_df_train['fake']
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf.predict(X_val)
roc_score = roc_auc_score(y_val, y_pred)
print(f'AUC-ROC Score: {roc_score:.2f}')

# Save the trained model
joblib.dump(rf, 'trained_model_1.pkl')