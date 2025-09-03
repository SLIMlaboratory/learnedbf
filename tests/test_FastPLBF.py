import unittest
import numpy as np
from learnedbf import FastPLBF
from learnedbf.classifiers import ScoredRandomForestClassifier, ScoredMLP, \
    ScoredDecisionTreeClassifier, ScoredLinearSVC


class TestFastPLBF(unittest.TestCase):
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
        # print('set the pseudo-random seed to 42')

    # def setUp(self):
        cls.filters = [
            FastPLBF(
                epsilon=0.01, N=200,
                classifier=ScoredDecisionTreeClassifier()),
            FastPLBF(
                epsilon=0.01, N=200,
                classifier=ScoredMLP(max_iter=100000, activation='logistic')),
            FastPLBF(epsilon=0.01, N=200,
                classifier=ScoredRandomForestClassifier()),
            # FastPLBF(epsilon=0.01, N=50,
            #     classifier=ScoredLinearSVC(max_iter=100_000, tol=0.1, C=.1))
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
        for plbf in TestFastPLBF.filters:
            assert plbf.is_fitted_

        
    def test_FN(self):
        for plbf in TestFastPLBF.filters:
            self.assertTrue(sum(plbf.predict(TestFastPLBF.objects[~TestFastPLBF.labels]) == 0))

    def test_FP(self):
        nonkeys = TestFastPLBF.objects[~TestFastPLBF.labels]
        for plbf in self.filters:
            fpr = plbf.estimate_FPR(nonkeys)
            self.assertAlmostEqual(fpr, 0.01, delta=0.01)

if __name__ == '__main__':
    unittest.main()
