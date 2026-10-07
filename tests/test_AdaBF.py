import unittest
import numpy as np
from learnedbf import AdaBF
from learnedbf.classifiers import ScoredRandomForestClassifier, ScoredMLP, \
    ScoredDecisionTreeClassifier, ScoredLinearSVC

from sklearn.exceptions import NotFittedError


class TestAdaBF(unittest.TestCase):

    def flip_bits(self, bit_mask, prob=0.1):
        mask = np.random.rand(bit_mask.shape[0]) > prob
        n_flipped = len(mask) - sum(mask)
        flipped_array = np.array([bit_mask[i] != 0 if p_ > prob \
                                   else not bit_mask[i]    
                                   for i,p_ in enumerate(mask)])
        return flipped_array, n_flipped
    
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        # print('set the pseudo-random seed to 42')

    def setUp(self):
        self.filters = [
            AdaBF(
                m=100000, 
                classifier=ScoredDecisionTreeClassifier(max_depth=3)),
            AdaBF(
                m=100000, 
                classifier=ScoredMLP(hidden_layer_sizes=(10,), max_iter=100000, activation='logistic')),
            AdaBF(m=500000, 
                classifier=ScoredRandomForestClassifier(n_estimators=10, max_depth=3)),
            AdaBF(m=100000, 
                classifier=ScoredLinearSVC(max_iter=100000, tol=0.1, C=0.1))
        ]

        n_samples = 100
        Fn = 0.1
        Fp = 0.1
        self.objects = np.expand_dims(np.arange(0, n_samples*2), axis=1)
        labels_f, _ = self.flip_bits(np.array([False] * n_samples), Fn)
        labels_t, _ = self.flip_bits(np.array([True] * n_samples), Fp)
        self.labels = np.concatenate((labels_f, labels_t))

        for adabf in self.filters:
            adabf.fit(self.objects, self.labels)
        

    def test_fit(self):
        for adabf in self.filters:
            assert adabf.is_fitted_

        
    def test_FN(self):
        for adabf in self.filters:
            self.assertTrue(sum(adabf.predict(self.objects[~self.labels]) == 0))

    def test_to_json_requires_fit(self):
        with self.assertRaises(NotFittedError):
            AdaBF().to_json()

    def test_json_round_trip_with_backup_filter(self):
        objects = np.expand_dims(np.arange(1, 10), axis=1)
        labels = [True, False, False, False, False, False, True, True, True]

        classifier = ScoredLinearSVC(random_state=522812,
                                     max_iter=100000,
                                     tol=0.1,
                                     C=0.1)
        classifier.fit(objects, labels)

        filter = AdaBF(classifier=classifier, m=10_000, n=len(objects))
        filter.fit(objects, labels)

        self.assertIsNotNone(filter.backup_filter_)

        filter_repr = filter.to_json()
        restored_filter = AdaBF(m=10, n=5)
        restored_filter.from_json(filter_repr)

        self.assertEqual(filter_repr, restored_filter.to_json())
        np.testing.assert_array_equal(restored_filter.predict(objects),
                                      filter.predict(objects))


if __name__ == '__main__':
    unittest.main()
