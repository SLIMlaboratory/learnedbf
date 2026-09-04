import math
import numpy as np
import unittest

from learnedbf.BF import ClassicalBloomFilter


class TestPybloomliveClassicalClassicalBloomFilter(unittest.TestCase):
        
    def test_train(self):
        for num_keys in np.logspace(1, 5, 5).astype(int):
            for epsilon in (0.01, 0.05, 0.1, 0.2):
                X = np.random.randint(0, 1_000_000, size=(num_keys, 1))
                bf = ClassicalBloomFilter(n=num_keys,
                                 epsilon=epsilon)
                bf.fit(X)
                self.assertTrue(bf.predict(X).all())

    def _check_empty(self, filter_class):
        bf = ClassicalBloomFilter(n=20,
                         epsilon=0.01,
                         filter_class=filter_class)

        with self.assertRaises(ValueError):
            bf.fit([[]])

    def test_empty_dataset(self):
        bf = ClassicalBloomFilter(n=20,
                         epsilon=0.01)

        with self.assertRaises(ValueError):
            bf.fit([[]])
        

    def test_score(self):
        for num_keys in np.logspace(1, 5, 5).astype(int):
            for epsilon in (0.01, 0.05, 0.1, 0.2):
                X = np.random.randint(0, 1_000_000, size=(num_keys, 1))
                bf = ClassicalBloomFilter(n=num_keys,
                                 epsilon=epsilon)
                bf.fit(X)

                num_queries = 10_000
                Q = np.random.randint(0, 1_000_000, size=(num_queries, 1))
                Q = np.expand_dims(
                    np.array(list(set(Q.squeeze()) - set(X.squeeze()))),
                    axis=1
                )

                # TODO: this is ugly, check with test set in LBF
                self.assertTrue(bf.estimate_FPR(Q) <= epsilon * 2)

    def test_param_consistency(self):
        n_values = (10, 100, 1000, 10000)
        epsilon_values = (0.01, 0.05, 0.1, 0.2)
        m_values = (200, 2000, 20000)
        for n in n_values:
            for epsilon in epsilon_values:
                bf = ClassicalBloomFilter(n=n, epsilon=epsilon)
                m = math.ceil(-n * np.log(epsilon) / np.log(2)**2)
                self.assertTrue(abs(bf.m - m) <= 1)

        for epsilon in epsilon_values:
            for m in m_values:
                bf = ClassicalBloomFilter(epsilon=epsilon, m=m)
                n = math.ceil(-m * np.log(2)**2 / np.log(epsilon))
                self.assertTrue(abs(bf.n - n) <= 1)


        n_m_values = ((10, 100), (10, 1000), (100, 2000), (1000, 10000))
        for n, m in n_m_values:
            bf = ClassicalBloomFilter(n=n, m=m)
            epsilon = np.exp(-np.log(2)**2 * m / n)
            if epsilon < 1e-9:
                epsilon = 1e-9
            self.assertEqual(bf.epsilon, epsilon)
        
        with self.assertRaises(ValueError):
            ClassicalBloomFilter(1, 1, 1)
            ClassicalBloomFilter()

    def test_export(self):
        for num_keys in np.logspace(1, 5, 5).astype(int):
            for epsilon in (0.01, 0.05, 0.1, 0.2):
                X = np.random.randint(0, 1_000_000, size=(num_keys, 1))
                bf = ClassicalBloomFilter(n=num_keys,
                                 epsilon=epsilon)
                bf.fit(X)

                bf_repr = bf.to_json()
                bf2 = ClassicalBloomFilter(n=23, epsilon=0.13)
                bf2.from_json(bf_repr)

                self.assertTrue(bf_repr == bf2.to_json())
                self.assertTrue(bf == bf2)

if __name__ == '__main__':
    unittest.main()
