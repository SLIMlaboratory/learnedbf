import unittest
import numpy as np
from learnedbf import PLBF
from learnedbf.classifiers import ScoredRandomForestClassifier, ScoredMLP, \
    ScoredDecisionTreeClassifier, ScoredLinearSVC


class TestPLBF_M(unittest.TestCase):

    @classmethod
    def flip_bits(cls, bit_mask, prob=0.1):
        # mask = np.random.rand(bit_mask.shape[0]) > prob

        # bit_mask[~mask] = ~bit_mask[~mask]

        # n_flipped = len(mask) - sum(mask)
        # flipped_array = np.array([bit_mask[i] != 0 if p_ > prob \
        #                            else not bit_mask[i]    
        #                            for i,p_ in enumerate(mask)])
        # return flipped_array, n_flipped
        return [not b if np.random.rand() <= prob else b for b in bit_mask]
    
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)

    # def setUp(self):
        cls.target_M = 100000.
        cls.filters = [
            PLBF(
                m=cls.target_M, N=50,
                classifier=ScoredDecisionTreeClassifier(max_depth=3)),
            PLBF(
                m=cls.target_M, N=50,
                classifier=ScoredMLP(hidden_layer_sizes=(10,), max_iter=100000, activation='logistic')),
            PLBF(m=500000., N=50,
                classifier=ScoredRandomForestClassifier(n_estimators=10, max_depth=3)),
        ]

        n_samples = 500
        Fn = 0.1
        Fp = 0.1
        cls.objects = np.expand_dims(np.arange(0, n_samples*2), axis=1)

        labels_f = cls.flip_bits(np.array([False] * n_samples), Fn)
        labels_t = cls.flip_bits(np.array([True] * n_samples), Fp)
        cls.labels = np.concatenate((labels_f, labels_t))
        # print(f'generated {sum(cls.labels)} key and {sum(~cls.labels)} non-keys')

        for plbf in cls.filters:
            plbf.fit(cls.objects, cls.labels)

    def test_fit(self):
        for plbf in TestPLBF_M.filters:
            assert plbf.is_fitted_

        
    def test_FN(self):
        for plbf in TestPLBF_M.filters:
            self.assertTrue(sum(plbf.predict(TestPLBF_M.objects[~TestPLBF_M.labels]) == 0))

    def test_M(self):
        nonkeys = TestPLBF_M.objects[~TestPLBF_M.labels]
        for plbf in self.filters:
            fpr = plbf.estimate_FPR(nonkeys)
            m = plbf.splbf.memory_usage_of_backup_bf
            # print(f'size (from memory_usage_of_backup_bf): {m}')
            # print(f'FP rate: {fpr}')
            # delta increased to account for classifier size being subtracted from m
            self.assertAlmostEqual(m, plbf.m, delta=100)

            m = plbf.get_size()['backup_filters']
            # print(f'size (from get_size): {m}')
            self.assertAlmostEqual(m, plbf.m, delta=100)

        

if __name__ == '__main__':
    unittest.main()
