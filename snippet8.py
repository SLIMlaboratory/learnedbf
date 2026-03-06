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


X, y = make_classification(n_samples=20000, n_features=2, n_redundant=0,
                           class_sep=0.5)
y = y.astype(bool)                           
X_build, X_evaluate, y_build, y_evaluate = train_test_split(X, y, test_size=0.1)

neg_indices = (y_evaluate == False)
X_build = np.vstack([X_build, X_evaluate[~neg_indices]])
y_build = np.hstack([y_build, y_evaluate[~neg_indices]])

X_evaluate = X_evaluate[neg_indices]
y_evaluate = y_evaluate[neg_indices]
assert np.all(~y_evaluate)

mlp = ScoredMLP()
mlp.fit(X_build, y_build)

names = ['LBF', 'SLBF', 'AdaBF', 'PLBF', 'FastPLBF', 'FastPLBFpp']
filters = [lbf.LBF, lbf.SLBF, lbf.AdaBF, lbf.PLBF, lbf.FastPLBF, lbf.FastPLBFpp]

epsilon = 0.01
print(f'Filters built with target epsilon={epsilon}')
print(42 * '-')
print('Filter       | eps   | size   | time')
print(42 * '-')

for n, f in zip(names, filters):
    if n == 'AdaBF':
        print('AdaBF        |   N/A |    N/A |    N/A')
    else:
        filter = f(epsilon=epsilon, classifier=mlp, threshold_test_size = 0.2)
        filter.fit(X_build, y_build)
        start = time.perf_counter()
        eps_hat = filter.estimate_FPR(X_evaluate[y_evaluate==0])
        end = time.perf_counter()
        time_taken = (end - start) / len(X_evaluate[y_evaluate==0])
        size = filter.get_size()
        m = sum([size[k] for k in size])
        print(f"{n:12} | {eps_hat:.3f} | {m:>6} | {time_taken:.2e} s")

print()
m = 51000
print(f'Filters built with target m={m}')
print(42 * '-')
print('Filter       | eps   | size   | time')
print(42 * '-')

for n, f in zip(names, filters):
    filter = f(m=m, classifier=mlp, threshold_test_size = 0.2)
    filter.fit(X_build, y_build)
    start = time.perf_counter()
    eps_hat = filter.estimate_FPR(X_evaluate[y_evaluate==0])
    end = time.perf_counter()
    time_taken = (end - start) / len(X_evaluate[y_evaluate==0])
    size = filter.get_size()
    m_actual = sum([size[k] for k in size])
    print(f"{n:12} | {eps_hat:.3f} | {m_actual:>6} | {time_taken:.2e} s")