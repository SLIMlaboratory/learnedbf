import unittest
import numpy as np
from learnedbf import FastPLBFpp
from learnedbf.classifiers import ScoredRandomForestClassifier, ScoredMLP, \
    ScoredDecisionTreeClassifier, ScoredLinearSVC


class TestFastPLBFpp_M(unittest.TestCase):

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
        cls.target_M = 2000.
        cls.filters = [
            FastPLBFpp(
                m=cls.target_M, N=50,
                classifier=ScoredDecisionTreeClassifier()),
            FastPLBFpp(
                m=cls.target_M, N=50,
                classifier=ScoredMLP(max_iter=100000, activation='logistic')),
            FastPLBFpp(m=cls.target_M, N=50,
                classifier=ScoredRandomForestClassifier()),
        ]

        n_samples = 500
        Fn = 0.1
        Fp = 0.1
        cls.objects = np.expand_dims(np.arange(0, n_samples*2), axis=1)

        labels_f = cls.flip_bits(np.array([False] * n_samples), Fn)
        labels_t = cls.flip_bits(np.array([True] * n_samples), Fp)
        cls.labels = np.concatenate((labels_f, labels_t))
        # print(f'generated {sum(cls.labels)} key and {sum(~cls.labels)} non-keys')

        for fastplbf in cls.filters:
            fastplbf.fit(cls.objects, cls.labels)

    def test_fit(self):
        for fastplbf in TestFastPLBFpp_M.filters:
            assert fastplbf.is_fitted_

        
    def test_FN(self):
        for fastplbf in TestFastPLBFpp_M.filters:
            self.assertTrue(sum(fastplbf.predict(TestFastPLBFpp_M.objects[~TestFastPLBFpp_M.labels]) == 0))

    def test_M(self):
        nonkeys = TestFastPLBFpp_M.objects[~TestFastPLBFpp_M.labels]
        for fastplbf in self.filters:
            fpr = fastplbf.estimate_FPR(nonkeys)
            m = fastplbf.splbf.memory_usage_of_backup_bf
            # print(f'size (from memory_usage_of_backup_bf): {m}')
            # print(f'FP rate: {fpr}')
            self.assertAlmostEqual(m, TestFastPLBFpp_M.target_M, delta=10)

            m = fastplbf.get_size()['backup_filters']
            # print(f'size (from get_size): {m}')
            self.assertAlmostEqual(m, TestFastPLBFpp_M.target_M, delta=10)

if __name__ == '__main__':
    unittest.main()
