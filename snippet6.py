import csv
import numpy as np
import learnedbf as lbf
from learnedbf import complexity_measures as cpl
from learnedbf.classifiers import ScoredMLP
import pandas as pd
from sklearn.model_selection import train_test_split 

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
m = sum([size[k] for k in size])

print(f"SLBF: epsilon={eps_hat:.3f}, size {m} bits")

classical_lbf = lbf.ClassicalBloomFilter(epsilon=epsilon, n=len(X[y]))
classical_lbf.fit(X[y])

print(f"Space gain w.r.t. classical BF: {m / classical_lbf.get_size():.2f}x")
