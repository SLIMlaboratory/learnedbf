import numpy as np
import math
import unittest
from learnedbf.classifiers import ScoredRandomForestClassifier, get_tree_size
from learnedbf import LBF, SLBF

learned_filter_classes = [LBF]

class TestScoredRF(unittest.TestCase):

    def test_to_json(self):
        objects = np.expand_dims(np.arange(1, 10), axis=1)
        labels = [False] * 6 + [True] * 3

        scl = ScoredRandomForestClassifier(random_state=42, n_estimators=3)
        scl.fit(objects, labels)

        
        lbf_repr = scl.to_json()
        self.assertTrue(True)

    def test_consistent_backup_filter(self):
        from_ = 1

        for to in [5, 10, 100, 10000]:
            mid = to * 2 // 3 # last negative item

            for num_FN in (1, 3, 10, 100, 1000):
                if num_FN >= mid:
                    break

                objects = np.expand_dims(np.arange(from_, to), axis=1)

                labels = [False] * (mid - from_ + 1) +  [True] * (to - mid - 1)

                labels[:num_FN] = [True] * num_FN

                scl = ScoredRandomForestClassifier(
                    random_state=42)
                scl.fit(objects, labels)

                for filter_class in learned_filter_classes:
                    lbf = filter_class(classifier=scl,
                                            epsilon=0.1,
                                            n=len(objects))
                    lbf.fit(objects, labels)

                    #self.assertIsNotNone(lbf.backup_filter_)

                    self.assertTrue(not lbf.predict(objects[:mid]).all())
                    self.assertTrue(lbf.predict(objects[mid:]).all())

    def test_consistent_initial_filter(self):

        n_samples = [50,100]
        Fp = 0.2

        for n in n_samples:
            objects = np.expand_dims(np.arange(0, n*2), axis=1)
            labels_f = np.array([False] * n)
            labels_t = np.array([True] * n)
            #we force a fraction of FP
            FP_mask = np.random.random(size=labels_t.shape[0]) < Fp
            labels_t[FP_mask] = ~labels_t[FP_mask]
            labels = np.concatenate((labels_f, labels_t))

            scl = ScoredRandomForestClassifier(max_depth=2)
            scl.fit(objects, labels)
            slbf = SLBF(epsilon=0.1)
            slbf.fit(objects, labels)

            self.assertIsNotNone(slbf.initial_filter_)
            self.assertEqual(sum(labels), slbf.initial_filter_.n)

    def test_get_size(self):
        n = 100

        for dim in [1, 3, 5, 10, 35]:
            objects = np.random.random(size=(n, dim))

            # we force the dataset to be linearly separable in the first dimension
            objects[:, 0] = np.arange(0, n)
            labels = [False] * int(math.floor(0.4 * n)) +  \
                    [True] * int(math.ceil(0.6 * n))

            rfcl = ScoredRandomForestClassifier(random_state=522812,
                                    n_estimators=3, min_samples_leaf=3, min_samples_split=4)
            rfcl.fit(objects, labels)

            for filter_class in learned_filter_classes:
                lbf = filter_class(classifier=rfcl,
                                   epsilon=0.1,
                                   n=len(objects))
                lbf.fit(objects, labels)

                self.assertTrue(rfcl.get_size() == sum([get_tree_size(t) for t in rfcl.estimators_]))

                lbf_size = rfcl.get_size()
                if lbf.backup_filter_ is not None:
                    lbf_size += lbf.backup_filter_.get_size()

                self.assertTrue(sum(lbf.get_size().values()) == lbf_size)
                self.assertTrue(rfcl.get_size() == lbf.classifier.get_size())
 


if __name__ == '__main__':
    unittest.main()
