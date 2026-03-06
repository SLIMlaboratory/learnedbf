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
import time

mlp = ScoredMLP()
filter = lbf.LBF(epsilon=0.01, classifier=mlp, threshold_test_size=0.2)
filter.fit(X_build, y_build)

mlp = ScoredMLP()
filter = lbf.LBF(epsilon=0.01, classifier=mlp,
                 threshold_test_size=0.2,
                 hyperparameters={'learning_rate_init':[0.01, 0.005, 0.001, 0.0005]})
filter.fit(X_build, y_build)

print(f"FPR:{filter.estimate_FPR(X_evaluate[y_evaluate==0]):.3f}")
pred = filter.predict(X_evaluate)