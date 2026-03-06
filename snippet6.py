import csv
import learnedbf as lbf
from learnedbf import complexity_measures as cpl
from learnedbf.classifiers import ScoredLinearSVC, ScoredMLP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 

X, y = make_classification(n_samples=20000, n_features=2, n_redundant=0,
                           class_sep=0.5)
y = y.astype(bool)                           
X_build, X_evaluate, y_build, y_evaluate = train_test_split(X, y, test_size=0.1)
X_train, X_test, y_train, y_test = train_test_split(X_build, y_build, test_size=0.1)

svc = ScoredLinearSVC()
svc.fit(X_train, y_train)

mlp = ScoredMLP()
mlp.fit(X_train, y_train)

threshold = 0.65

svc_pred = (svc.predict_score(X_test) > threshold).astype(int)
mlp_pred = (mlp.predict_score(X_test) > threshold).astype(int)

svc_score = accuracy_score(y_test, svc_pred)
mlp_score = accuracy_score(y_test, mlp_pred)

print(f'SVC score = {svc_score:.2f}, MLP score = {mlp_score:.2f}')
