import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from learnedbf import *

class TestFullyLearnedBloomFilter(unittest.TestCase):     

    def _flip_bits(self, bit_mask, prob=0.1):
        mask = np.random.rand(bit_mask.shape[0]) > prob
        n_flipped = len(mask) - sum(mask)
        flipped_array = np.array([bit_mask[i] != 0 if p_ > prob \
                                   else not bit_mask[i]    
                                   for i,p_ in enumerate(mask)])
        return flipped_array, n_flipped

    def setUp(self):
        self.not_fitted_flbf = FLBF(epsilon=0.1)

        self.filters = [
            FLBF(epsilon=0.1, t=10, verbose=False),
            FLBF(epsilon=0.05, t=10)
        ]

        n_samples = 2000
        X, y = make_classification(
            n_samples=n_samples, n_features=2, n_informative=2, 
            n_redundant=0, n_clusters_per_class=1, weights=[0.8, 0.2], 
            class_sep=.8, random_state=42)
        scaler = StandardScaler()
        self.objects = scaler.fit_transform(X)
        self.labels = np.array([True if i else False for i in y])

        for flbf in self.filters:
            flbf.fit(self.objects, self.labels)   

    def test_fit(self):
        for flbf in self.filters:
            assert flbf.is_fitted_

    def test_score(self):
        self.assertRaises(ValueError, self.not_fitted_flbf.score, self.objects, self.labels)

        for flbf in self.filters:
            self.assertTrue(abs(flbf.estimate_FPR(self.objects, self.labels) - flbf.epsilon) <= flbf.epsilon)

    def test_size(self):
        self.assertRaises(NotFittedError, self.not_fitted_flbf.get_size)

        for flbf in self.filters:
            for size in flbf.get_size():
                self.assertGreater(size, 0)

if __name__ == '__main__':
    unittest.main()
