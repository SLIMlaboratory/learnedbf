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

f1v = cpl.F1v()
fig, ax = plt.subplots()

compl = []
sep = np.linspace(0.001, 1.5, 10)
for s in sep:
    X, y = make_classification(n_samples=20000, n_features=2, n_redundant=0,
                               class_sep=s)
    c = f1v.compute(X, y)
    compl.append(c)
ax.plot(sep, compl)
plt.show()
