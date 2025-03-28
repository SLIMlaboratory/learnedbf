import abc
import logging
from random import randint
import mmh3
import math
import time
import numpy as np
from bitarray import bitarray
import pybloom_live
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin

def check_params_(n, epsilon, m):
    """Check the consistency of the arguments passed to the constructor.

    :param n: number of keys, defaults to `None`.
    :type n: `int`
    :param epsilon: expected false positive rate, defaults to `None`.
    :type epsilon: `float`
    :param m: size (in bits) of the bitmap, defaults to `None`.
    :type m: `int`

    :return: a tuple (n, epsilon, m) containing the values of the three
        parameters once consistency has been verified or enforced.
    :type: `tuple`

    Note: the three parameters are linked by the following relation:

    .. math::
    m = -(n \\cdot \\log epsilon) / ((\\log 2)^2)

    thus if they are all specified, the function will verify that this
    relation holds, rasing ValueError otherwise; when two out of three
    parameters are specified, the unspecified one will be obtained by
    enforcing this relation; in all other cases, a ValueError will be
    raised.


    """
    num_specified = sum([x is not None for x in (n, epsilon, m)])
    if num_specified == 3:
        if abs(m * np.log(2)**2 + n * np.log(epsilon)) < 10e-6:
            return n, epsilon, m
        else:
            raise ValueError(f'n={n!r}, epsilon={epsilon!r} and '
                             f'm={m!r} are not consistent for the '
                             'creation of a Bloom filter')

    if num_specified != 2:
        raise ValueError('At least two values between n, epsilon, and m '
                         'have to be specified for creating a Bloom filter'
                         f' (provided n={n!r}, epsilon={epsilon!r}, '
                         f'm={m!r})')

    if n is None:
        return -m * np.log(2)**2 / np.log(epsilon), epsilon, m

    if epsilon is None:
        guessed_epsilon = np.exp(-np.log(2)**2 * m / n)
        if guessed_epsilon > 1:
            raise ValueError(f'Inconsistent values (n={n!r}, m={m!r}), '
                             'leading to FPR > 1')
        if guessed_epsilon < 1e-9:
            guessed_epsilon = 1e-9
        return n, guessed_epsilon, m

    if m is None:
        return n, epsilon, -n * np.log(epsilon) / np.log(2)**2

class BloomFilter:
    """Base BloomFilter class."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, n=None, epsilon=None):
        """Create an instance of :class:`BloomFilter`.

        :param n: number of keys, defaults to 10.
        :type n: `int`
        :param epsilon: expected false positive rate, defaults to 0.1.
        :type epsilon: `float`
        """

        self.n = n
        self.epsilon = epsilon

    @abc.abstractclassmethod
    def fit(self, X, y=None):
        """Build the Bloom Filter. Abstract method implemented in
        subclasses.
        """

        return

    @abc.abstractclassmethod
    def predict(self, X):
        """Computes predictions for a set of queries, each to be checked
        for inclusion in the Bloom Filter. Abstract method implemented
        in subclasses.
        """

        return

    def estimate_FPR(self, X, y=None):
        """Compute the empirical false positive rate of the Bloom Filter
        on a set of queries.

        :param X: queries to be checked.
        :type X: array of numerical arrays
        :param y: labels for the elements in `X`, defaults to `None`.
        :type y: array of `bool`
        :return: empirical false positive rate.
        :rtype: `float`

        Note: the rate is computed only on non-key values. Thus, if y is
        provided, all true labels (that is, keys) are removed; otherwise,
        all elements in X are assumed to be non-key values.
        """

        check_is_fitted(self, 'is_fitted_')
        if y is not None:
            X, y = check_X_y(X, y)
            X = X[~y]
            X = check_array(X)
        else:
            X = check_array(X)

        return self.predict(X).sum() / len(X)

    @abc.abstractclassmethod
    def get_size(self):
        """Return the Bloom Filter size (in bits). Abstract method implemented in
        subclasses.
        """

        return



class ClassicalBloomFilterImpl():

    def __init__(self, n=None, epsilon=None, m=None):
        """
        :param n: number of keys.
        :type n: `int`
        :param epsilon: expected false positive rate.
        :type epsilon: `float`
        """

        self.n, self.epsilon, self.m = check_params_(n, epsilon, m)
        
        # Size of bit array to use
        self.m = self.get_size()

        # number of hash functions to use
        self.k = self.get_hash_count(self.m, self.n)

        # Bit array of given size
        self.bit_array = bitarray(self.m)

        # initialize all bits as 0
        self.bit_array.setall(0)

    def add(self, item):
        '''
        Add an item in the filter
        '''
        digests = []
        for i in range(self.k):
            # i works as seed to mmh3.hash() function
            digest = mmh3.hash(item, i) % self.m
            digests.append(digest)

            # set the bit True in bit_array
            self.bit_array[digest] = True

    def check(self, item):
        '''
        Check for existence of an item in filter
        '''
        for i in range(self.k):
            digest = mmh3.hash(item, i) % self.m
            if not self.bit_array[digest]:
                # if any of bit is False then,its not present
                # in filter
                # else there is probability that it exist
                return False
        return True

    def get_size(self):
        return int(self.m)

    @classmethod
    def get_hash_count(self, m, n):
        '''
        Return the hash function(k) to be used using
        following formula
        k = (m/n) *)

        m : int
            size of(k)ar
        n : int
            number (k)ems exped to be stored in filter
        '''

        k = (m/n) * math.log(2)
        return int(k)
    
class hashfunc():

    def __init__(self, m):
        self.m = m
        self.seed = randint(1, 99999999)

    def __call__(self, x):
        return mmh3.hash(x, self.seed) % self.m

class VarhashBloomFilter(BaseEstimator, ClassifierMixin, BloomFilter):

    def __init__(self, m, k_max):
        """Ada-BF implementation

        This code is adapted from the repository associated with the paper "Adaptive Learned Bloom Filter (Ada-BF)" 
        Repository link: https://github.com/DAIZHENWEI/Ada-BF

        :param m: the length of the bit array.
        :type key: `int`
        :param k_max: the maximum number of hash function to use.
        :type k_max: `int`
        """
        self.k_max = k_max 
        self.m = int(m)
        self.h = [hashfunc(self.m) for _ in range(int(k_max))]
        self.bit_array = bitarray(self.m)
        self.bit_array.setall(0)

    def add(self, key, k):
        """Adds a key to the filter using a specified number of hash functions.

        :param key: the key to add.
        :type key: `int`
        :param k: the number of hash function to use.
        :type k: `int`
        """
        for hf in self.h[:int(k)]:
             self.bit_array[hf(key)] = True

    def check(self, key, k):
        """Test the key membership using a specified number of hash functions.

        :param key: the key to be checked.
        :type key: `int`
        :param k: the number of hash function to use.
        :type k: `int`
        """
        test_result = False
        match = 0

        for hf in self.h[:int(k)]:
            match += 1*(self.bit_array[hf(key)] == 1) 
        if match == k:
            test_result = True
        return test_result
    
    def fit(self, X, y=None, K=None):
        """Build the Bloom Filter.

        :param X: array containing keys.
        :type X: array of numbers
        :param y: array containing labels for X's elements, defaults to
            `None`.
        :type y: array of `bool`
        :param K: array containing the number of hash functions to be used 
            for each element, defaults to`None`.
        :type K: array of `int`
        :return: the fit Varhash Bloom Filter instance.
        :rtype: :class:`VarhashBloomFilter`
        :raises: `ValueError` if `X` is empty.
        :raises: `ValueError` if `K` is `None` or `X` and `K` lengths don't match.
        """

        if y is None:
            y = [True] * len(X)

        X, y = check_X_y(X, y)

        if (~y).any():
            logging.warn('non-keys were removed')
            X = X[y]
            y = y[y]

        if len(X) == 0:
            raise ValueError('Empty set of keys')
        
        if K is None or len(K) != len(X):
            raise ValueError(f'Inconsistent K size passed, '
                             f'required {len(X)!r}')
        
        #Se passo una lista python trasformo in ndarray
        K = np.array(K)
        if K.dtype not in ['int32', 'int64']:
            raise ValueError(f'Inconsistent K type passed {len(K)!r}, '
                             f'required array of `int`')

        for x, k in zip(X, K):
            self.add(x, k)

        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X, K=None):
        """Computes predictions for a set of queries, each to be checked
        for inclusion in the Bloom Filter.

        :param X: queries to be checked.
        :type X: array of numerical arrays
        :param K: array containing the number of hash functions to be used 
            for each element, defaults to`None`.
        :type K: array of `int`
        :return: prediction for each value in `X`.
        :rtype: array of `bool`
        :raises: `ValueError` if `K` is `None` or `X` and `K` lengths don't match.
        """

        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        if K is None or len(K) != len(X):
            raise ValueError(f'Inconsistent K size passed, '
                             f'required {len(X)!r}')
        
        K = np.array(K)
        if K.dtype not in ['int32', 'int64']:
            raise ValueError(f'Inconsistent K type passed {len(K)!r}, '
                            f'required array of `int`')


        return np.array([self.check(x, k) for x,k in zip(X, K)])

    def get_size(self):
        return int(self.m)


class PyBloomLiveAdapter:
    """Adapter allowing to use objects of
    :class:`pybloom_live.BloomFilter` using the same interface of
    :class:`pybloom.BloomFilter`."""

    def __init__(self, n=None, epsilon=None, m=None):
        """Create an instance of the adapter.

        :param n: number of keys, defaults to None.
        :type n: `int`
        :param epsilon: expected false positive rate, defaults to None.
        :type epsilon: `float`
        :param m: size (in bits) of the bitmap, defaults to None.
        :type m: `int`

        Note: if two out of the three parameters are specified, the remaining
        one will be obtained accordingly. If all the values are specified,
        their consistency will be verified, throwing a ValueError in case of
        failure. In all other cases, a ValueError will be thrown.
        """

        self.n, self.epsilon, self.m = check_params_(n, epsilon, m)
        self.filter = pybloom_live.BloomFilter(self.n, self.epsilon)

    def __repr__(self):
        return f'PyBloomLiveAdapter(n={self.n!r}, epsilon={self.epsilon!r})'

    def add(self, x):
        """Delegate `add` method to :class:`pybloom_live.BloomFilter`

        :param x: key to be added to the filter.
        :type x: numerical array
        """

        self.filter.add(x)

    def check(self, x):
        """Implements `check` of :class:`pybloom.BloomFilter` interface
        in terms of the implementation of operator `in` in
        :class:`pybloom_live.BloomFilter`.


        :param x: query to be checked for inclusion in the filter.
        :type x: numerical array
        :return: True if the query is predicted to be in the filter,
            False otherwise.
        :type: `bool`
        """

        return x in self.filter

    def get_size(self):
        """Return the size of the Bloom Filter (in bits).

        :return: size of the bit array (in bits)
        :rtype: `int`
        """

        return int(self.m) + 1


class ClassicalBloomFilter(BloomFilter, BaseEstimator, ClassifierMixin):
    """Implementation of a classical Bloom Filter."""

    def __init__(self, n=None, epsilon=None, m=None,
                 filter_class=ClassicalBloomFilterImpl):
        """Create an instance of :class:`ClassicalBloomFilter`.

        :param n: number of keys, defaults to `None`.
        :type n: `int`
        :param epsilon: expected false positive rate, defaults to `None`.
        :type epsilon: `float`
        :param m: size of the bitmap (in bits), defaults to `None`.
        :type m: `int`

        :param filter_class: used Bloom Filter class, defaults to
            :class:'ClassicalBloomFilterImpl'.
        :type filter_class: `class`
        """

        self.filter_class = filter_class
        self.bf_ = filter_class(n, epsilon, m)
        self.n, self.epsilon, self.m = self.bf_.n, self.bf_.epsilon, self.bf_.m

    def __repr__(self):
        args = []
        args.append(f'n={self.n!r}')
        args.append(f'epsilon={self.epsilon!r}')
        if self.filter_class != ClassicalBloomFilter:
            args.append(f'filter_class={self.filter_class!r}')

        args = ', '.join(args)
        return f'BloomFilter({args})'
    
    def add(self, x):
        """Delegate `add` method to the bloom filter implementation

        :param x: key to be added to the filter.
        """
        self.bf_.add(x)

    def check(self, x):
        """Delegate `check` method to the bloom filter implementation

        :param x: key to be added to the filter.
        """
        return self.bf_.check(x)
    

    def fit(self, X, y=None):
        """Build the Bloom Filter.

        :param X: array containing keys.
        :type X: array of numbers
        :param y: array containing labels for X's elements, defaults to
            `None`.
        :type y: array of `bool`
        :return: the fit Bloom Filter instance.
        :rtype: :class:`BloomFilter`
        :raises: `ValueError` if `X` is empty.
        """

        if y is None:
            y = [True] * len(X)

        X, y = check_X_y(X, y)

        if (~y).any():
            logging.warn('non-keys were removed')
            X = X[y]
            y = y[y]

        if len(X) == 0:
            raise ValueError('Empty set of keys')

        if len(X) != self.n:
            raise ValueError(f'Inconsistent set of keys: passed {len(X)!r}, '
                             f'required {self.n!r}')

        for x in X:
            self.add(x)

        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Computes predictions for a set of queries, each to be checked
        for inclusion in the Bloom Filter.

        :param X: queries to be checked.
        :type X: array of numerical arrays
        :return: prediction for each value in `X`.
        :rtype: array of `bool`
        """

        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        return np.array([self.check(x) for x in X])

    def get_size(self):
        """Return the Bloom Filter size.

        :return: dictionary describing the overall size of the Bloom Filter,
            in which the key `'bitmap'` is associated to the number of bits
            required by the bitmap of the filter.
        :rtype: `dict`
        """

        return self.bf_.get_size()


def run_BF(FPR, key_set, testing_list):
    '''
    Crea un BF con key-set = key_set e target fpr = FPR
    e ritorna FPR empirico, dimensione del BF e tempo di accesso medio per
    elemento calcolati testando la struttura su testing_list.
    '''

    BF = BloomFilter(len(key_set), FPR)
    for url in key_set:
        BF.add(url)

    fps = 0
    total = 0
    total_time = 0

    for urlt in testing_list:
        total += 1
        start = time.time()
        result = BF.check(urlt)
        end = time.time()
        total_time += (end-start)
        if result:
            fps += 1

    avg_fp = fps/total
    # print(f"avg fp : {fps/total} , fps :{fps}, total: {total},
    #         {BF.check(testing_list[2])}")

    # returns empirical FPR, BF size in bytes, and access time per element
    return avg_fp, BF.size/8, (total_time)/len(testing_list)
