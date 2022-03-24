#!/usr/bin/env python
# coding: utf-8

# Import libraries and requirements

import sys
import subprocess

print('Verifying or installing dependencies')
try:
    import flair
except:
    print('Installing flair library')
    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'flair'])

try:
    import numpy
except:
    print('Installing numpy library')
    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy'])

try:
    import pandas
except:
    print('Installing pandas library')
    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])

try:
    import scikit_learn
except:
    print('Installing scikit_learn library')
    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit_learn'])

try:
    import tqdm
except:
    print('Installing tqdm library')
    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tqdm'])

try:
    import fasttext
except:
    print('Installing fasttext library')
    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'fasttext'])

try:
    import gensim
except:
    print('Installing fasttext library')
    # implement pip as a subprocess:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gensim'])

##try:
##    import allennlp
##except:
##    print('Installing allennlp library')
##    # implement pip as a subprocess:
##    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'allennlp==0.9.0'])


# Loading required packages
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

import fasttext
import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')  # English



### Run scripts and train models for each of the word embeddings
import ELMo_Text_Classification_Rev_dataset
import BERT_Text_Classification_Rev
import ELMo_Text_Classification_Rev
import fasttext_Text_Classification_Rev
import Flair_Text_Classification_Rev
import GloVe_Text_Classification_Rev
import Word2vec_Text_Classification_Rev

