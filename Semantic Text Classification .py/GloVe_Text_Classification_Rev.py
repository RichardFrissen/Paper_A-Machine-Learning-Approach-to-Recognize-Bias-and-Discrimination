#!/usr/bin/env python
# coding: utf-8

# Import libraries and requirements
import pandas as pd
from numpy import mean
from numpy import std
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.dummy import DummyClassifier
from statistics import mean
import json
import csv
import ast
import numpy as np

from flair.embeddings import WordEmbeddings
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings
from tqdm import tqdm
import pickle
import time
from datetime import datetime


import fasttext
import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')


# Set seed
seed = np.random.seed(1)

timenow = datetime.now().strftime('%c')
print('Starting glove script. Starting at {}\n'.format(timenow))

# Read dataset containing the full EMSCAD dataset including linguistic features
data = pd.read_csv('data_features_full_wordembedding_glove_complete.csv', na_values=['nan'])


# Replace NaN values with a "0"
data = data.replace(np.nan, '0', regex=True)


# We drop the token, as it is no longer needed for prediction
data.drop('Token', axis=1, inplace=True)


# 80% / 20% split
# Train, Test = train_test_split(data1, test_size=0.2, shuffle=False)

X = data.drop(['Label'],axis=1).values # independant features
y = data['Label'].values # dependant variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# Delete data to save memory
del(data)


# Set max_iter and create empty lists for the results
max_iterations = 1000000000

classifier = []
accuracy = []
precision = []
recall = []
f1 = []

cv_classifier = []
cv_precision = []
cv_recall = []
cv_f1 = []

##### Model training - Baseline
##
##timenow = datetime.now().strftime('%c')
##print('Glove word embedding - Start baseline. Starting at {}\n'.format(timenow))
##
##clf = DummyClassifier(strategy="uniform", random_state=seed)
##
##
### Model fit
##clf.fit(X_train, y_train)
##
##
### Save model - Baseline
##f = open('glove_baseline.pckl', 'wb')
##pickle.dump(clf, f)
##f.close()
##
### Evaluation Baseline
##
##y_pred = clf.predict(X_test)
##
##print("Accuracy:", accuracy_score(y_test, y_pred))
##print("Precision:", precision_score(y_test, y_pred, average='macro',zero_division=0))
##print("Recall:", recall_score(y_test, y_pred, average='macro'))
##print("F1_score:", f1_score(y_test, y_pred, average='macro'))
##
##classifier.append("Baseline")
##accuracy.append(accuracy_score(y_test, y_pred))
##precision.append(precision_score(y_test, y_pred, average='macro',zero_division=0))
##recall.append(recall_score(y_test, y_pred, average='macro'))
##f1.append(f1_score(y_test, y_pred, average='macro'))
##
##
##### Model training - Logisitc Regression
##
##timenow = datetime.now().strftime('%c')
##print('Glove word embedding - Start LR. Starting at {}\n'.format(timenow))
##
##scoring = ['precision_macro', 'recall_macro', "f1_macro"]
##clf = LogisticRegression(solver='newton-cg', random_state=seed, max_iter=max_iterations)
##scores_LR = cross_validate(clf, X_train, y_train, scoring = scoring, cv=10, n_jobs=-1)
##LR_avg_precision = mean(scores_LR['test_precision_macro'])
##LR_avg_recall = mean(scores_LR['test_recall_macro'])
##LR_avg_f1 = mean(scores_LR['test_f1_macro'])
##
##cv_classifier.append("LR")
##cv_precision.append(LR_avg_precision)
##cv_recall.append(LR_avg_recall)
##cv_f1.append(LR_avg_f1)
##
### Model fit
##clf.fit(X_train, y_train)
##
##
### Save model - Logistic Regression
##f = open('glove_lr.pckl', 'wb')
##pickle.dump(clf, f)
##f.close()
##
### Evaluation Logistic Regression
##
##y_pred = clf.predict(X_test)
##
##print("Accuracy:", accuracy_score(y_test, y_pred))
##print("Precision:", precision_score(y_test, y_pred, average='macro',zero_division=0))
##print("Recall:", recall_score(y_test, y_pred, average='macro'))
##print("F1_score:", f1_score(y_test, y_pred, average='macro'))
##
##classifier.append("LR")
##accuracy.append(accuracy_score(y_test, y_pred))
##precision.append(precision_score(y_test, y_pred, average='macro',zero_division=0))
##recall.append(recall_score(y_test, y_pred, average='macro'))
##f1.append(f1_score(y_test, y_pred, average='macro'))


### Model training - Decision Tree

timenow = datetime.now().strftime('%c')
print('Glove word embedding - Start DT. Starting at {}\n'.format(timenow))

scoring = ['precision_macro', 'recall_macro', "f1_macro"]
clf = DecisionTreeClassifier(random_state=seed)
scores_DT = cross_validate(clf, X_train, y_train, scoring = scoring, cv=10, n_jobs=-1)
DT_avg_precision = mean(scores_DT['test_precision_macro'])
DT_avg_recall = mean(scores_DT['test_recall_macro'])
DT_avg_f1 = mean(scores_DT['test_f1_macro'])

cv_classifier.append("DT")
cv_precision.append(DT_avg_precision)
cv_recall.append(DT_avg_recall)
cv_f1.append(DT_avg_f1)

# Model fit
clf.fit(X_train, y_train)


# Save model - Decision Tree
f = open('glove_dt.pckl', 'wb')
pickle.dump(clf, f)
f.close()

# Evaluation Decision Tree

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro',zero_division=0))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1_score:", f1_score(y_test, y_pred, average='macro'))

classifier.append("DT")
accuracy.append(accuracy_score(y_test, y_pred))
precision.append(precision_score(y_test, y_pred, average='macro',zero_division=0))
recall.append(recall_score(y_test, y_pred, average='macro'))
f1.append(f1_score(y_test, y_pred, average='macro'))


### Model training - Naive Bayes

timenow = datetime.now().strftime('%c')
print('Glove word embedding - Start NB. Starting at {}\n'.format(timenow))

scoring = ['precision_macro', 'recall_macro', "f1_macro"]
clf = GaussianNB()
scores_NB = cross_validate(clf, X_train, y_train, scoring = scoring, cv=10, n_jobs=-1)
NB_avg_precision = mean(scores_NB['test_precision_macro'])
NB_avg_recall = mean(scores_NB['test_recall_macro'])
NB_avg_f1 = mean(scores_NB['test_f1_macro'])

cv_classifier.append("NB")
cv_precision.append(NB_avg_precision)
cv_recall.append(NB_avg_recall)
cv_f1.append(NB_avg_f1)

# Model fit
clf.fit(X_train, y_train)


# Save model - Naive Bayes
f = open('glove_nb.pckl', 'wb')
pickle.dump(clf, f)
f.close()

# Evaluation Naive Bayes

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro',zero_division=0))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1_score:", f1_score(y_test, y_pred, average='macro'))

classifier.append("NB")
accuracy.append(accuracy_score(y_test, y_pred))
precision.append(precision_score(y_test, y_pred, average='macro',zero_division=0))
recall.append(recall_score(y_test, y_pred, average='macro'))
f1.append(f1_score(y_test, y_pred, average='macro'))


### Dump and save results

#Cross-validation results
results_cv = pd.DataFrame(zip(cv_classifier, cv_precision, cv_recall, cv_f1), columns = ['CV_Classifier', 'CV_Precision', 'CV_Recall', 'CV_F1-score'])
results_cv = results_cv.sort_values(by = "CV_F1-score", ascending = False)

f = open('glove_cv_results.pckl', 'wb')
pickle.dump(results_cv, f)
f.close()

#Model fit results
results = pd.DataFrame(zip(classifier, accuracy, precision, recall, f1), columns = ['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1-score'])
results = results.sort_values(by = "F1-score", ascending = False)

f = open('glove_results.pckl', 'wb')
pickle.dump(results, f)
f.close()

f = open('glove_results.pckl', 'rb')
results = pickle.load(f)
f.close()

# Save results dataframe
results.to_csv('glove_results.csv', index = False)
results_cv.to_csv('glove_cv_results.csv', index = False)


