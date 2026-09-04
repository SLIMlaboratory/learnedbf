import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from learnedbf import LBF
from learnedbf.classifiers import ScoredLinearSVC
from learnedbf.classifiers import ScoredDecisionTreeClassifier
from learnedbf.classifiers import ScoredRandomForestClassifier


class TestLBF(unittest.TestCase):

    def test_to_json_requires_fit(self):
        with self.assertRaises(NotFittedError):
            LBF().to_json()


    def test_json_round_trip_without_backup_filter(self):
        classifiers = [
            ScoredLinearSVC(random_state=42, max_iter=100000, tol=0.1),
            ScoredDecisionTreeClassifier(random_state=42, max_depth=3),
            ScoredRandomForestClassifier(random_state=42, n_estimators=5),
            ]

        objects = np.expand_dims(np.arange(1, 10), axis=1)
        labels = [False] * 6 + [True] * 3

        for classifier in classifiers:
            classifier.fit(objects, labels)

            lbf = LBF(classifier=classifier, epsilon=0.1, n=len(objects))
            lbf.fit(objects, labels)

            self.assertIsNone(lbf.backup_filter_)

            lbf_repr = lbf.to_json()
            restored_lbf = LBF()
            restored_lbf.from_json(lbf_repr)

            self.assertEqual(lbf_repr, restored_lbf.to_json())
            np.testing.assert_array_equal(restored_lbf.predict(objects),
                                        lbf.predict(objects))

    def test_json_round_trip_with_backup_filter(self):
        objects = np.expand_dims(np.arange(1, 10), axis=1)
        labels = [True, False, False, False, False, False, True, True, True]

        classifier = ScoredLinearSVC(random_state=522812,
                                     max_iter=100000,
                                     tol=0.1,
                                     C=0.1)
        classifier.fit(objects, labels)

        lbf = LBF(classifier=classifier, epsilon=0.1, n=len(objects))
        lbf.fit(objects, labels)

        self.assertIsNotNone(lbf.backup_filter_)

        lbf_repr = lbf.to_json()
        restored_lbf = LBF()
        restored_lbf.from_json(lbf_repr)

        self.assertEqual(lbf_repr, restored_lbf.to_json())
        np.testing.assert_array_equal(restored_lbf.predict(objects),
                                      lbf.predict(objects))


if __name__ == '__main__':
    unittest.main()