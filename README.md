# learnedbf


<p align="center">
<a href="https://www.python.org"><img alt="Python version" src="https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white"></a>
<a href="https://github.com/SLIMlaboratory/learnedbf/actions"><img alt="Actions Status" src="https://github.com/psf/black/workflows/Test/badge.svg"></a>
<a href="https://learnedbf.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/learnedbf/badge/?version=latest"></a>
<a href="https://pypi.org/project/learnedbf/"><img alt="PyPI" src="https://img.shields.io/pypi/v/learnedbf"></a>
<a href="https://github.com/SLIMlaboratory/learnedbf/blob/main/LICENSE"><img alt="License: Apache2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>

</p>

A python package for Learned Bloom Filters

`learnedbf` is a Python package for Learned Bloom Filters (LBF), intended as
[Bloom Filters](https://en.wikipedia.org/wiki/Bloom_filter) learned from
data, as originally proposed by
[Kraska et al., 2018](https://arxiv.org/abs/1712.01208).

This page provides a quick start guide. For more comprehensive information,
please refer to the [documentation](https://learned.readthedocs.io/en/latest/).

## Installation

```shell
pip install learnedbf
```

## Usage

### Library import

The following code imports all libraries used in the subsequent snippets.

```pycon
>>> import numpy as np
>>> 
>>> from sklearn.datasets import make_classification
>>> from sklearn.metrics import accuracy_score
>>> from sklearn.model_selection import train_test_split 
>>> 
>>> import learnedbf as lbf
>>> from learnedbf.classifiers import ScoredLinearSVC, ScoredMLP
>>> from learnedbf import complexity_measures as cpl
```

### Evaluating the complexity of a dataset

The following code generates datasets of decrerasing complexity using the
`make_classification` function available in Scikit-learn, evaluating for each
the corresponding F1v measure.

```pycon
>>> f1v = cpl.F1v()
>>>
>>> sep = np.linspace(0.001, 1.5, 10)
>>> for s in sep:
...     X, y = make_classification(n_samples=20000, n_features=2, n_redundant=0,
...                                class_sep=s)
...     c = f1v.compute(X, y)
...     print(f'class separation {s:.2f}, F1V {c:.2f}')
...
class separation 0.00, F1V 1.00
class separation 0.17, F1V 0.86
class separation 0.33, F1V 0.59
class separation 0.50, F1V 0.33
class separation 0.67, F1V 0.27
class separation 0.83, F1V 0.20
class separation 1.00, F1V 0.11
class separation 1.17, F1V 0.11
class separation 1.33, F1V 0.10
class separation 1.50, F1V 0.08
```

### Training classifiers

The following code generates the dataset used in the rest of the examples,
dividing it in three splits. The first two ones will be used for training a
classifier and evaluating its performance; the third one will be used afterwards
to estimate the FPR of the built filters.

```pycon
>>> X, y = make_classification(n_samples=20000, n_features=2, n_redundant=0,
>>>                            class_sep=0.5)
>>> y = y.astype(bool)                           
>>> X_build, X_evaluate, y_build, y_evaluate = train_test_split(X, y,
...                                                             test_size=0.1)
>>> X_train, X_test, y_train, y_test = train_test_split(X_build, y_build,
...                                                     test_size=0.1)
```

The following code trains a linear SVC and a multi-layer perceptron, comparing
their performance on the test set using accuracy.

```pycon
>>> svc = ScoredLinearSVC()
>>> svc.fit(X_train, y_train)
>>> 
>>> mlp = ScoredMLP()
>>> mlp.fit(X_train, y_train)
>>> 
>>> threshold = 0.65
>>> 
>>> svc_pred = (svc.predict_score(X_test) > threshold).astype(int)
>>> mlp_pred = (mlp.predict_score(X_test) > threshold).astype(int)
>>> 
>>> svc_score = accuracy_score(y_test, svc_pred)
>>> mlp_score = accuracy_score(y_test, mlp_pred)
>>> 
>>> print(f'SVC score = {svc_score:.2f}, MLP score = {mlp_score:.2f}')
SVC score = 0.56, MLP score = 0.85
```

### Building a learned Bloom filter

The following code builds a LBF using the previously learned MLP, and estimates
its empirical FPR.

```pycon
>>> filter = lbf.LBF(epsilon=0.01, classifier=mlp, threshold_test_size = 0.2)
>>> filter.fit(X_build, y_build)
LBF(epsilon=0.01, classifier=ScoredMLP(), threshold=0.7520010828581488)
>>> print(f"FPR:{filter.estimate_FPR(X_evaluate[y_evaluate==0]):.3f}")
FPR:0.009
```

The following code builds a LBF backed by a multi-layer perceptron, now training
the latter on the provided data.

```pycon
>>> filter = lbf.LBF(epsilon=0.01, classifier=mlp, threshold_test_size = 0.2)
>>> filter.fit(X_build, y_build)
LBF(epsilon=0.01, classifier=ScoredMLP(), threshold=0.7520010828581488)
>>> mlp = ScoredMLP()
>>> filter = lbf.LBF(epsilon=0.01, classifier=mlp, threshold_test_size=0.2)
>>> filter.fit(X_build, y_build)
LBF(epsilon=0.01, classifier=ScoredMLP(), threshold=0.7239171235297011)
>>> print(f"FPR:{filter.estimate_FPR(X_evaluate[y_evaluate==0]):.3f}")
FPR:0.009
```

The following code repeats the previous operation, now also performing a model
selection on the learning rate of the multi-layer perceptron.

```pycon
>>> filter = lbf.LBF(epsilon=0.01, classifier=mlp, threshold_test_size = 0.2)
>>> filter.fit(X_build, y_build)
>>> 
>>> mlp = ScoredMLP()
>>> filter = lbf.LBF(epsilon=0.01, classifier=mlp,
...                  threshold_test_size=0.2,
...                  hyperparameters={
...                      'learning_rate_init':[0.01, 0.005, 0.001, 0.0005]})
>>> filter.fit(X_build, y_build)
LBF(epsilon=0.01, classifier=ScoredMLP(), hyperparameters={'learning_rate_init': [0.01, 0.005, 0.001, 0.0005]}, threshold=0.741263017250332)
>>> print(f"FPR:{filter.estimate_FPR(X_evaluate[y_evaluate==0]):.3f}")
FPR:0.009
```

## License

[Apache 2.0](https://github.com/SLIMlaboratory/learnedbf/blob/master/LICENSE).



## Authors

`learnedbf` has been designed and implemented by D. Malchiodi, M. Frasca, N. Rinaldi and R. Giancarlo.

