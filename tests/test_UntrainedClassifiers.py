import unittest
import numpy as np
from learnedbf import *
from learnedbf.classifiers import ScoredRandomForestClassifier, ScoredMLP, ScoredDecisionTreeClassifier, ScoredLinearSVC
from sklearn.neighbors import KNeighborsClassifier


class TestLBF2(unittest.TestCase):
    def setUp(self):
        self.lbf = LBF()

        self.filters = [
            LBF(classifier=ScoredDecisionTreeClassifier(), epsilon=0.1),
            LBF(classifier=ScoredMLP(max_iter=100000), epsilon=0.1),
            LBF(classifier=ScoredRandomForestClassifier(), epsilon=0.1),
            LBF(classifier=ScoredDecisionTreeClassifier(max_depth=3), m=100000),
            LBF(classifier=ScoredMLP(hidden_layer_sizes=(10,), max_iter=100000), m=100000),
            LBF(classifier=ScoredRandomForestClassifier(n_estimators=10, max_depth=3), m=500000)
        ]

        # self.n = 100

        # X = np.random.randint(low=0, high=1000, size=self.n)
        # y = X > 500
        # X = np.expand_dims(X, axis=1)

        # X_train, self.X_test, y_train, self.y_test = train_test_split(X, y)
        # n_train = sum(y_train)

        num_keys = 1000
        X = np.random.randint(0, 1000000, size=(2*num_keys, 1))
        y_true = (X > 500000) | (X < 20000)
        X = np.expand_dims(X, axis = 1)
        X_neg_train, X_neg_test = train_test_split(X[~y_true])

        X_train = np.vstack((X[y_true], X_neg_train))
        y_train = np.array([True]*len(X[y_true]) + [False] * len(X_neg_train))

        self.X_test = np.vstack((X[y_true], X_neg_test))
        self.y_test = np.array([True]*len(X[y_true]) + [False] * len(X_neg_test))

        for lbf in self.filters:
             assert lbf.fit(X_train, y_train)

    def test_fit(self):
        for lbf in self.filters:
             assert lbf.is_fitted_

    def test_score(self):

        self.assertRaises(ValueError, self.lbf.score, self.X_test, self.y_test)

        for lbf in self.filters[:3]:
            self.assertTrue(abs(lbf.estimate_FPR( \
                self.X_test, self.y_test) - lbf.epsilon) <= 1E-1)
            
    def test_FN(self):

        for lbf in self.filters:
            self.assertTrue(sum(lbf.predict(self.X_test[self.y_test]) == 0) == 0)

if __name__ == '__main__':
    unittest.main()
