import unittest
import numpy as np
from learnedbf import *
from learnedbf.classifiers import ScoredRandomForestClassifier, ScoredMLP, \
    ScoredDecisionTreeClassifier, ScoredLinearSVC


class TestSanswichedLearnedBloomFilter(unittest.TestCase):

    def flip_bits(self, bit_mask, prob=0.1):
        mask = np.random.rand(bit_mask.shape[0]) > prob
        n_flipped = len(mask) - sum(mask)
        flipped_array = np.array([bit_mask[i] != 0 if p_ > prob \
                                   else not bit_mask[i]    
                                   for i,p_ in enumerate(mask)])
        return flipped_array, n_flipped

    def setUp(self):
        self.lbf = SLBF(epsilon=0.1)

        self.filters = [
            SLBF(
                epsilon=0.1, 
                classifier=ScoredDecisionTreeClassifier()),
            SLBF(
                epsilon=0.1, 
                classifier=ScoredMLP(max_iter=100000, activation='logistic')),
            SLBF(epsilon=0.1, 
                classifier=ScoredRandomForestClassifier()),
            SLBF(epsilon=0.1, 
                classifier=ScoredLinearSVC(max_iter=100000, tol=0.1, C=0.1)),
            SLBF(
                m=100000, 
                classifier=ScoredDecisionTreeClassifier(max_depth=3)),
            SLBF(
                m=100000, 
                classifier=ScoredMLP(hidden_layer_sizes=(10,), max_iter=100000, activation='logistic')),
            SLBF(m=500000, 
                classifier=ScoredRandomForestClassifier(n_estimators=10, max_depth=3)),
            SLBF(m=100000, 
                classifier=ScoredLinearSVC(max_iter=100000, tol=0.1, C=0.1))
        ]

        # self.n = 100

        # X = np.random.randint(low=0, high=1000, size=self.n)
        # y = X > 500
        # X = np.expand_dims(X, axis=1)

        # X_train, self.objects, y_train, self.labels = train_test_split(X, y)
        # n_train = sum(y_train)

        n_samples = 100
        Fn = 0.1
        Fp = 0.1
        self.objects = np.expand_dims(np.arange(0, n_samples*2), axis=1)
        labels_f, n_Fn = self.flip_bits(np.array([False] * n_samples), Fn)
        labels_t, n_Fp = self.flip_bits(np.array([True] * n_samples), Fp)
        self.labels = np.concatenate((labels_f, labels_t))

        for slbf in self.filters:
            slbf.fit(self.objects, self.labels)
        

    def test_fit(self):
        for slbf in self.filters:
            assert slbf.is_fitted_

    def test_score(self):
        self.assertRaises(ValueError, self.lbf.score, self.objects, self.labels)

        for slbf in self.filters[:4]:
            self.assertTrue(abs(slbf.estimate_FPR( \
                self.objects, self.labels) - slbf.epsilon) <= slbf.epsilon)
        
    def test_FN(self):
        for slbf in self.filters:
            self.assertTrue(sum(slbf.predict(self.objects[~self.labels]) == 0))

    def test_json_round_trip_without_backup_filter(self):
            classifiers = [
                ScoredLinearSVC(random_state=42, max_iter=100000, tol=0.1),
                ScoredDecisionTreeClassifier(random_state=42, max_depth=3),
                ScoredRandomForestClassifier(random_state=42, n_estimators=5),
                ScoredMLP(hidden_layer_sizes=(3,), random_state=42, max_iter=100000, tol=0.1),
                ]
    
            objects = np.expand_dims(np.arange(1, 10), axis=1)
            labels = [False] * 6 + [True] * 3
    
            for classifier in classifiers:
                classifier.fit(objects, labels)
    
                filter = SLBF(classifier=classifier, epsilon=0.1, n=len(objects))
                filter.fit(objects, labels)
    
                self.assertIsNone(filter.lbf_.backup_filter_)
    
                filter_repr = filter.to_json()
                restored_filter = SLBF()
                restored_filter.from_json(filter_repr)
    
                self.assertEqual(filter_repr, restored_filter.to_json())
                np.testing.assert_array_equal(restored_filter.predict(objects),
                                            filter.predict(objects))
    
    def test_json_round_trip_with_backup_filter(self):
        objects = np.expand_dims(np.arange(1, 10), axis=1)
        labels = [True, False, False, False, False, False, True, True, True]

        classifier = ScoredLinearSVC(random_state=522812,
                                        max_iter=100000,
                                        tol=0.1,
                                        C=0.1)
        classifier.fit(objects, labels)

        filter = SLBF(classifier=classifier, epsilon=0.1, n=len(objects))
        filter.fit(objects, labels)

        self.assertIsNotNone(filter.lbf_.backup_filter_)

        filter_repr = filter.to_json()
        restored_filter = SLBF()
        restored_filter.from_json(filter_repr)

        self.assertEqual(filter_repr, restored_filter.to_json())
        np.testing.assert_array_equal(restored_filter.predict(objects),
                                        filter.predict(objects))

if __name__ == '__main__':
    unittest.main()
