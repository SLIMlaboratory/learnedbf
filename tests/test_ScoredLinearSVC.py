import numpy as np
import unittest
from learnedbf.classifiers import ScoredLinearSVC
from learnedbf.BF import BloomFilter
from learnedbf import LBF, SLBF

learned_filter_classes = [LBF]

class TestScoredLinearSVC(unittest.TestCase):

    def test_linear_separability(self):
        from_ = 1

        for to in [5, 10, 100]:
            mid = to * 2 // 3 # last negative item

            objects = np.expand_dims(np.arange(from_, to), axis=1)

            labels = [False] * (mid - from_ + 1) +  [True] * (to - mid - 1)


            scl = ScoredLinearSVC(random_state=42,
                              max_iter=100000, tol=0.1)
            scl.fit(objects, labels)

            for filter_class in learned_filter_classes:

                lbf = filter_class(classifier=scl,
                                   epsilon=0.1,
                                   n=len(objects))
                lbf.fit(objects, labels)

                self.assertIsNone(lbf.backup_filter_)

                self.assertFalse(lbf.predict(objects[:mid]).all())
                self.assertTrue(lbf.predict(objects[mid:]).all())

                # it is pointless to test the correctness of the classifier,
                # as if the backup filter is not needed (and we check this)
                # the classifier cannot have made any error.


    def test_consistent_backup_filter(self):
        from_ = 1

        for to in [5, 10, 100]:
            mid = to * 2 // 3 # last negative item

            for num_FN in (1, 3, 10, 100, 1000):
                if num_FN >= mid:
                    break

                objects = np.expand_dims(np.arange(from_, to), axis=1)

                labels = [False] * (mid - from_ + 1) +  [True] * (to - mid - 1)

                labels[:num_FN] = [True] * num_FN

                scl = ScoredLinearSVC(random_state=522812,
                                    max_iter=100000, tol=0.1, C=0.1)
                scl.fit(objects, labels)

                for filter_class in learned_filter_classes:
                    lbf = filter_class(classifier=scl,
                                       epsilon=0.1,
                                       n=len(objects))
                    lbf.fit(objects, labels)

                    self.assertIsNotNone(lbf.backup_filter_)

                    self.assertFalse(lbf.predict(objects[:mid]).all())
                    self.assertTrue(lbf.predict(objects[mid:]).all())

                    self.assertTrue(lbf.backup_filter_.n == num_FN)

    def test_consistent_initial_filter(self):

        n_samples = [20, 50]
        Fp = 0.2

        for n in n_samples:
            objects = np.expand_dims(np.arange(0, n*2), axis=1)
            labels_f = np.array([False] * n)
            labels_t = np.array([True] * n)
            #we force a fraction of FP
            FP_mask = np.random.random(size=labels_t.shape[0]) < Fp
            labels_t[FP_mask] = ~labels_t[FP_mask]
            labels = np.concatenate((labels_f, labels_t))

            scl = ScoredLinearSVC(max_iter=100000, tol=0.1, C=0.1)
            scl.fit(objects, labels)
            slbf = SLBF(epsilon=0.1)
            slbf.fit(objects, labels)

            self.assertIsNotNone(slbf.initial_filter_)
            self.assertEqual(sum(labels), slbf.initial_filter_.n)

    def test_get_size(self):
        n = 10

        for dim in [1, 3, 5, 10, 35]:
            objects = np.random.random(size=(n, dim))

            # we force the dataset to be linearly separable in the first dimension
            objects[:, 0] = np.arange(0, n)
            labels = [False] * 4 +  [True] * 6

            scl = ScoredLinearSVC(random_state=522812,
                                    max_iter=100000, tol=0.1, C=0.1)
            scl.fit(objects, labels)

            for filter_class in learned_filter_classes:
                lbf = filter_class(classifier=scl,
                                   epsilon=0.1,
                                   n=len(objects))
                lbf.fit(objects, labels)

                self.assertTrue(scl.get_size() == (dim+1) * scl.float_size)

                lbf_size = scl.get_size()
                if lbf.backup_filter_ is not None:
                    lbf_size += lbf.backup_filter_.get_size()

                self.assertTrue(sum(lbf.get_size().values()) == lbf_size)

                self.assertTrue(scl.get_size() == lbf.classifier.get_size())

    def test_export(self):
        n = 10

        for dim in [1, 3, 5, 10, 35]:
            np.random.seed(seed=42)
            objects = np.random.random(size=(n, dim))

            # we force the dataset to be linearly separable in the first dimension
            objects[:, 0] = np.arange(0, n)
            labels = [False] * 4 +  [True] * 6

            scl = ScoredLinearSVC(random_state=522812,
                                    max_iter=100000, tol=0.1, C=0.1)
            scl.fit(objects, labels)
            scl_repr = scl.to_json()
            scl2 = ScoredLinearSVC().from_json(scl_repr)

            self.assertTrue(scl_repr == scl2.to_json())
            self.assertTrue(scl == scl2)
  
 


if __name__ == '__main__':
    unittest.main()

    # add tests for get_size in all classifiers, complete tests for DT e RF
