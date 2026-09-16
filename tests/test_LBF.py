import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from learnedbf import LBF
from learnedbf.classifiers import ScoredLinearSVC
from learnedbf.classifiers import ScoredDecisionTreeClassifier
from learnedbf.classifiers import ScoredRandomForestClassifier
from learnedbf.classifiers import ScoredMLP


class TestLBF(unittest.TestCase):

    def test_to_json_requires_fit(self):
        with self.assertRaises(NotFittedError):
            LBF().to_json()


    def test_json_round_trip_without_backup_filter(self):
        classifiers = [
            ScoredLinearSVC(random_state=42, max_iter=100000, tol=0.1),
            ScoredDecisionTreeClassifier(random_state=42, max_depth=3),
            ScoredRandomForestClassifier(random_state=42, n_estimators=5),
            ScoredMLP(hidden_layer_sizes=(3,), random_state=42, max_iter=100000, tol=0.1),
            ]

        objects = np.expand_dims(np.arange(1, 10), axis=1)
        labels = [False] * 6 + [True] * 3

        for classifier in classifiers:
            classifier.fit(objects, labels)

            filter = LBF(classifier=classifier, epsilon=0.1, n=len(objects))
            filter.fit(objects, labels)

            self.assertIsNone(filter.backup_filter_)

            filter_repr = filter.to_json()
            restored_filter = LBF()
            restored_filter.from_json(filter_repr)

            self.assertEqual(filter_repr, restored_filter.to_json())
            np.testing.assert_array_equal(restored_filter.predict(objects),
                                        filter.predict(objects))

    def test_json_round_trip_with_backup_filter(self):
        objects = np.expand_dims(np.arange(1, 10), axis=1)
        labels = [True, False, False, False, False, False, True, True, True]

        classifier = ScoredLinearSVC(random_state=522812,
                                     max_iter=100000,
                                     tol=0.1,
                                     C=0.1)
        classifier.fit(objects, labels)

        filter = LBF(classifier=classifier, epsilon=0.1, n=len(objects))
        filter.fit(objects, labels)

        self.assertIsNotNone(filter.backup_filter_)

        filter_repr = filter.to_json()
        restored_filter = LBF()
        restored_filter.from_json(filter_repr)

        self.assertEqual(filter_repr, restored_filter.to_json())
        np.testing.assert_array_equal(restored_filter.predict(objects),
                                      filter.predict(objects))


if __name__ == '__main__':
    unittest.main()