import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import learnedbf as lbf
from learnedbf.classifiers import ScoredMLP

X, y = make_classification(n_samples=50000, n_features=2, n_redundant=0,
                            class_sep=0.9, weights=[0.7, 0.3])


X_pos = X[y==1]
X_neg = X[y==0]
y_pos = y[y==1].astype(bool)
y_neg = y[y==0].astype(bool)

X_neg_build, X_neg_evaluate, y_neg_build, y_neg_evaluate = \
        train_test_split(X_neg, y_neg, test_size=0.5)

X_neg_train, X_neg_build, y_neg_train, y_neg_build = \
        train_test_split(X_neg_build, y_neg_build, test_size=0.5)

X_train = np.vstack([X_pos, X_neg_train])
y_train = np.hstack([y_pos, y_neg_train])

X_build = np.vstack([X_pos, X_neg_build])
y_build = np.hstack([y_pos, y_neg_build])

mlp = ScoredMLP(hidden_layer_sizes=(5, 5))
mlp.fit(X_train, y_train)

print('scores for positive train examples')
scores = mlp.predict_score(X_pos)
print(f'{min(scores)} -> {max(scores)}')
print('scores for negative train examples')
scores = mlp.predict_score(X_neg_train)
print(f'{min(scores)} -> {max(scores)}')

print('building filter with pretrained mlp')
filter_pretrained = lbf.LBF(epsilon=0.01, classifier=mlp,
                 threshold_test_size = 0.2,
                 num_candidate_thresholds=20)
filter_pretrained.fit(X_build, y_build)

assert mlp is filter_pretrained.classifier

print('When using a pretraned MLP')
print(f"FPR:{filter_pretrained.estimate_FPR(X_neg_evaluate):.3f}")

predictions = (filter_pretrained.classifier.predict_score(X_neg_evaluate) > filter_pretrained.threshold)
estimated_fpr_external = sum(predictions) / len(predictions)
print(f'externally estimated FPR = {estimated_fpr_external:.3f}')

print('building filter training mlp, no model selection')
mlp = ScoredMLP()
filter_untrained = lbf.LBF(epsilon=0.01, classifier=mlp,
                 threshold_test_size=0.2,
                 num_candidate_thresholds=5000)
filter_untrained.fit(np.vstack([X_build, X_neg_train]), 
                     np.hstack([y_build, y_neg_train]))
print('When training MLP, no model selection')
print(f"FPR:{filter_untrained.estimate_FPR(X_neg_evaluate):.3f}")


print('building filter reusing classifier of previous filter')
reused_mlp = filter_untrained.classifier
filter_reused = lbf.LBF(epsilon=0.01, classifier=reused_mlp,
                 threshold_test_size=0.2,
                 num_candidate_thresholds=5000)
filter_reused.fit(X_build, y_build)
print(f"FPR:{filter_reused.estimate_FPR(X_neg_evaluate):.3f}")


print('building filter training a mlp, with model selection')
mlp = ScoredMLP()
filter = lbf.LBF(epsilon=0.01, classifier=mlp,
                 threshold_test_size=0.2,
                 num_candidate_thresholds=5000,
                 hyperparameters={'learning_rate_init': [0.1, 0.01,
                                                         0.001, 0.0001]})
filter.fit(X_build, y_build)
print('When training MLP, with model selection')
print(f"FPR:{filter.estimate_FPR(X_neg_evaluate):.3f}")
pred = filter.predict(X_neg_evaluate)