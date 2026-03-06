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

with open('data/url_dataset_unique_features.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

df = pd.DataFrame(data[1:], columns=data[0])

X = df.drop(columns=['data', 'label']).astype(float).values
y = ((df.label.astype(int) + 1) / 2).astype(bool)

f1v_score = cpl.F1v().compute(X, y)
print(f"F1v score: {f1v_score:.4f}")

X_build, X_evaluate, y_build, y_evaluate = train_test_split(X, y, test_size=0.1)

neg_indices = (y_evaluate == False)
X_build = np.vstack([X_build, X_evaluate[~neg_indices]])
y_build = np.hstack([y_build, y_evaluate[~neg_indices]])

X_evaluate = X_evaluate[neg_indices]
y_evaluate = y_evaluate[neg_indices]

mlp = ScoredMLP(hidden_layer_sizes=(10,), max_iter=500)
mlp.fit(X_build, y_build)

epsilon = 0.01

filter = lbf.SLBF(epsilon=epsilon, classifier=mlp, threshold_test_size = 0.2)
filter.fit(X_build, y_build)
eps_hat = filter.estimate_FPR(X_evaluate[~y_evaluate])
size = filter.get_size()

m_classifier = size['classifier']
m_supp_filters = sum([size[k] for k in size if k != 'classifier'])
m = m_classifier + m_supp_filters

print(f"SLBF: epsilon={eps_hat:.3f}")
print(f"size (bits): {m_classifier} (classifier) + {m_supp_filters} (filters)")

classical_bf = lbf.ClassicalBloomFilter(epsilon=epsilon, n=len(X[y]))
classical_bf.fit(X[y])

print(f"Relative space of classical BF: {classical_bf.get_size() / (m):.2f}x")