import math
import numpy as np
import unittest

from learnedbf.BF import VarhashBloomFilter


class TestVarHashBloomFilter(unittest.TestCase):
        
    def test_train(self):
        for num_keys in np.logspace(1, 5, 5).astype(int):
            X = np.random.randint(0, 1_000_000, size=(num_keys, 1))

            #TODO multiple tests for values of m and k_max
            m = 10_000
            k_max = 10
            vbf = VarhashBloomFilter(m, k_max)
            K = [np.random.randint(k_max) for _ in X]
            vbf.fit(X, K=K)
            # for x, k in zip(X, K):
            #     vbf.add(x, k)
            
            self.assertTrue(vbf.predict(X, K).all())    

    def test_export(self):
        for num_keys in np.logspace(1, 5, 5).astype(int):
            X = np.random.randint(0, 1_000_000, size=(num_keys, 1))
            m = 10_000
            k_max = 10
            vbf = VarhashBloomFilter(m, k_max)
            K = [np.random.randint(k_max) for _ in X]
            vbf.fit(X, K=K)
            # for x, k in zip(X, K):
            #     vbf.add(x, k)


            vbf_repr = vbf.to_json()
            vbf2 = VarhashBloomFilter(m, k_max)
            vbf2.from_json(vbf_repr)

            self.assertTrue(vbf_repr == vbf2.to_json())
            self.assertTrue(vbf == vbf2)

if __name__ == '__main__':
    unittest.main()
