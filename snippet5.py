
import numpy as np
import learnedbf as lbf
from learnedbf.classifiers import ScoredMLP
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split 

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

names = ['LBF', 'SLBF', 'AdaBF',
         'PLBF',
         'FastPLBF', 'FastPLBFpp']
filters = [lbf.LBF, lbf.SLBF, lbf.AdaBF,
          lbf.PLBF,
          lbf.FastPLBF, lbf.FastPLBFpp]

epsilon = 0.01
print(f'Filters built with target epsilon={epsilon}')
print(29 * '-')
print('Filter       | eps   | size')
print(29 * '-')

for n, f in zip(names, filters):
    if n == 'AdaBF':
        print('AdaBF        |   N/A |    N/A')
    else:
        filter = f(epsilon=epsilon, classifier=mlp, threshold_test_size = 0.2)
        filter.fit(X_build, y_build)
        eps_hat = filter.estimate_FPR(X_evaluate[y_evaluate==0])
        size = filter.get_size()
        m = sum([size[k] for k in size])
        print(f"{n:12} | {eps_hat:.3f} | {m:>6}")

print()

m = 51000
print(f'Filters built with target m={m}')
print(29 * '-')
print('Filter       | eps   | size')
print(29 * '-')

for n, f in zip(names, filters):
    filter = f(m=m, classifier=mlp, threshold_test_size = 0.2)
    filter.fit(X_build, y_build)
    eps_hat = filter.estimate_FPR(X_evaluate[y_evaluate==0])
    size = filter.get_size()
    m_actual = sum([size[k] for k in size])
    print(f"{n:12} | {eps_hat:.3f} | {m_actual:>6}")
