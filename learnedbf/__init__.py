import abc
import gc
import logging
import math
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import auc, precision_recall_curve, make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, \
                                    train_test_split
from sklearn.utils.validation import NotFittedError, check_X_y, check_array, \
                                     check_is_fitted

from learnedbf.BF import BloomFilter, ClassicalBloomFilter, VarhashBloomFilter, ClassicalBloomFilterImpl
from learnedbf.classifiers import ScoredDecisionTreeClassifier

# TODO: check the behavior when using non-integer keys
# TODO: check what happens with the `classes_` attribute of classifiers
#       not based on trees

__version__ = '0.5.4e'

logging.getLogger(__name__).addHandler(logging.NullHandler())

def auprc(y, y_hat):
    precision, recall, thresholds = precision_recall_curve(y, y_hat)
    return auc(recall, precision)

def auprc_score(cls, X, y):
    scorer =  make_scorer(auprc)
    return scorer(cls, X, y)


def threshold_evaluate(epsilon, key_predictions, nonkey_predictions):
    epsilon_tau = nonkey_predictions.sum() / len(nonkey_predictions)
    if epsilon_tau >= epsilon:
        # epsilon_tau >= epsilon, constraint not met
        # return None, to get aware the caller that the
        # current candidate threshold should not be considered
        return None

    epsilon_b = (epsilon - epsilon_tau) / (1 - epsilon_tau)

    # compute m_b (backup filter bitmap size)
    num_fn = (~key_predictions).sum()
    m_b = -num_fn * np.log(epsilon_b) / np.log(2)**2
    return epsilon_tau, epsilon_b, m_b

def check_y(y):
    """ Check if the input array has valid labels for binary classification.
        Valid combinations are (False, True), (0, 1), (-1, 1)
    """
    if  y.dtype.type == np.bool_: return y

    if np.all(np.isin(y, [-1, 1])) or np.all(np.isin(y, [0, 1])):
        print("Warning: all the values of y will be casted to bool")
        return y == 1
    
    raise ValueError("Possible values for y are (0, 1), (-1, 1) or \
                        (False, True)")


class LBF(BaseEstimator, BloomFilter, ClassifierMixin):
    """Implementation of the Learned Bloom Filter"""

    # MASTER TODO: Each classifier class should
    # 1. output predictions in terms of True/False (MA VERO???)
    # 2. implement a get_size method returning the size in bits of the model
    # 3. implement a predict_score method returning the score of the
    #    classifier, intended as how confident the classifier is in saying that
    #    an element is a key.

    def __init__(self,
                 n=None,
                 epsilon=None,
                 m=None,
                 classifier=ScoredDecisionTreeClassifier(),
                 hyperparameters={},
                 num_candidate_thresholds=10,
                 threshold_test_size=0.7,
                 fpr_test_size=0.3,
                 model_selection_method=StratifiedKFold(n_splits=5,
                                                        shuffle=True),
                 scoring=auprc_score,
                 threshold_evaluate=threshold_evaluate,
                 threshold=None,
                 classical_BF_class=ClassicalBloomFilterImpl,
                 backup_filter_size=None,
                 random_state=4678913,
                 verbose=False):
        """Create an instance of :class:`LBF`.

        :param n: number of keys, defaults to None.
        :type n: `int`
        :param epsilon: expected false positive rate, defaults to None.
        :type epsilon: `float`
        :param m: dimension in bit of the bitmap, defaults to None.
        :type m: `int`
        :param classifier: classifier to be trained, defaults to
            :class:`DecisionTreeClassifier`.
        :type classifier: :class:`sklearn.BaseEstimator`
        :param hyperparameters: grid of values for hyperparameters to be
            considered during classifier training, defaults to {}.
        :type hyperparameters: `dict`
        :param num_candidate_thresholds: number of candidate thresholds to be
            considered for mapping classifier scores onto predictions,
            defaults to 10.
        :type num_candidate_thresholds: `int`
        :param threshold_test_size: relative validation set size used to set
            the best classifier threshold, defaults to 0.7.
        :param fpr_test_size: relative test set size used to estimate
            the empirical FPR of the learnt Bloom filter, defaults
            to 0.3.
        :type fpr_test_size: `float`
        :param model_selection_method: strategy to be used for
            discovering the best hyperparameter values for the learnt
            classifier, defaults to
            `StratifiedKFold(n_splits=5, shuffle=True)`.
        :type model_selection_method:
            :class:`sklearn.model_selection.BaseShuffleSplit` or
            :class:`sklearn.model_selection.BaseCrossValidator`
        :param scoring: method to be used for scoring learnt
            classifiers, defaults to `auprc`.
        :type scoring: `str` or function
        :param threshold_evaluate: function to be used to optimize the
          classifier threshold choice (NOTE: at the current implementation
          stage there are no alternatives w.r.t. minimizing the size of the
          backup filter).
        :type threshold_evaluate: function
        :param threshold: the threshold of the bloom filter, defaults to `None`.
        :type threshold: `float`
        :param classical_BF_class: class of the backup filter, defaults
            to :class:`ClassicalBloomFilterImpl`.
        :param backup_filter_size: the size of the backup filter, defaults to `None`.
        :type backup_filter_size: `int`
        :param random_state: random seed value, defaults to `None`,
            meaning the current seed value should be kept.
        :type random_state: `int` or `None`
        :param verbose: flag triggering verbose logging, defaults to
            `False`.
        :type verbose: `bool`
        """

        self.n = n
        self.epsilon = epsilon
        self.m = m
        self.classifier = classifier
        self.hyperparameters = hyperparameters
        self.num_candidate_thresholds = num_candidate_thresholds
        self.fpr_test_size = fpr_test_size
        self.threshold_test_size = threshold_test_size
        self.model_selection_method = model_selection_method
        self.scoring = scoring
        self.threshold_evaluate = threshold_evaluate
        self.threshold = threshold
        self.classical_BF_class = classical_BF_class
        self.backup_filter_size = backup_filter_size
        self.random_state = random_state
        self.verbose = verbose

    def __repr__(self):
        args = []
        if self.epsilon != None:
            args.append(f'epsilon={self.epsilon}')
        if self.classifier.__repr__() != 'ScoredDecisionTreeClassifier()':
            args.append(f'classifier={self.classifier}')
        if self.hyperparameters != {}:
            args.append(f'hyperparameters={self.hyperparameters}')
        if self.threshold != None:
            args.append(f'threshold={self.threshold}')
        if self.fpr_test_size != 0.3:
            args.append(f'fpr_test_size={self.fpr_test_size}')
        if (self.model_selection_method.__class__ != StratifiedKFold or 
            self.model_selection_method.n_splits != 5 or 
            self.model_selection_method.shuffle != True):
            args.append(f'model_selection_method={self.model_selection_method}')
        if callable(self.scoring):
            if self.scoring.__name__ != 'auprc_score':
                args.append(f'scoring={self.scoring.__name__}')
        else:
            score_name = args.append(f'scoring="{self.scoring}"')
        if self.random_state != 4678913:
            args.append(f'random_state={self.random_state}')
        if self.verbose != False:
            args.append(f'verbose={self.verbose}') 
        
        args = ', '.join(args)
        return f'LBF({args})'
    
    def fit(self, X, y):
        """Fits the Learned Bloom Filter, training its classifier,
        setting the score threshold and building the backup filter.

        :param X: examples to be used for fitting the classifier.
        :type X: array of numerical arrays
        :param y: labels of the examples.
        :type y: array of `bool`
        :return: the fit Bloom Filter instance.
        :rtype: :class:`LBF`
        :raises: `ValueError` if X is empty, or if no threshold value is
            compliant with the false positive rate requirements.

        NOTE: If the classifier variable instance has been specified as an
        already trained classifier, `X` and `y` are considered as the dataset
        to be used to build the LBF, that is, setting the threshold for the
        in output of the classifier, and evaluating the overall empirical FPR.
        In this case, `X` and `y` are assumed to contain values not used in
        order to infer the classifier, in order to ensure a fair estimate of
        FPR. Otherwise, `X` and `y` are meant to be the examples to be used to
        train the classifier, and subsequently set the threshold and evaluate
        the empirical FPR.
        """

        if self.m is None and self.epsilon is None:
            raise ValueError("At least one parameter \
                             between mand epsilon must be specified.")
            
        if len(X) == 0:
            raise ValueError('Empty set of keys')

        X, y = check_X_y(X, y)

        if self.random_state is not None:
            self.model_selection_method.random_state = self.random_state
        
        X_pos = X[y]

        self.n = len(X_pos)

        y_pos = y[y]

        try:
            check_is_fitted(self.classifier)
            # a trained classifier was passed to the constructor
            X_neg_threshold_test = X[~y]
        except NotFittedError:
            # the classifier has to be trained
            X_neg = X[~y]

            X_neg_trainval, X_neg_threshold_test = \
                train_test_split(X_neg,
                                 test_size=self.threshold_test_size,
                                 random_state=self.random_state)
            del X_neg
            gc.collect()
            y_neg_trainval = [False] * len(X_neg_trainval)
            X_trainval = np.vstack([X_pos, X_neg_trainval])
            del X_neg_trainval
            gc.collect()

            y_trainval = np.hstack([y_pos, y_neg_trainval])
            del y_neg_trainval
            gc.collect()

            model = GridSearchCV(estimator=self.classifier,
                                 param_grid=self.hyperparameters,
                                 cv=self.model_selection_method,
                                 scoring=self.scoring,
                                 refit=True,
                                 n_jobs=-1)
            model.fit(X_trainval, y_trainval)
            del X_trainval, y_trainval
            gc.collect()
            self.classifier = model.best_estimator_

        backup_filter_fpr = None
        if self.threshold is None:
            key_scores = self.classifier.predict_score(X_pos)
            del X_pos
            gc.collect()
            nonkey_scores = self.classifier.predict_score(X_neg_threshold_test)
            del X_neg_threshold_test
            gc.collect()

            scores = np.hstack([key_scores, nonkey_scores])
            candidate_threshold = \
                np.quantile(scores,
                            np.linspace(0,
                                        1 - 1 / len(scores),
                                        self.num_candidate_thresholds))

            self.backup_filter_size = np.inf
            self.threshold = None

            if self.m is not None:
                epsilon = 1
                self.backup_filter_size = self.m
                for t in candidate_threshold:
                    key_predictions = (key_scores > t)
                    nonkey_predictions = (nonkey_scores > t)

                    num_fn = (~key_predictions).sum()
                    Fp = nonkey_predictions.sum() / len(nonkey_predictions)
                       
                    if num_fn == 0:
                        #no FN so epsilon_lbf = epsilon_tau
                        epsilon_lbf = Fp
                        e_b=1

                    else:
                        e_b = np.exp(-np.log(2)**2 * self.m / num_fn)
                        if self.verbose:
                            print(f"e_b = {e_b}")
                        epsilon_lbf = Fp + (1-Fp) * e_b


                    if self.verbose:
                            print(f"t = {t}")
                            print(f"Fp = {Fp}")
                            print(f"nFn = {num_fn}")
                            print(f"bf epsilon={e_b}")
                            print(f"lbf epsilon={epsilon_lbf}")

                    if epsilon_lbf < epsilon:
                        backup_filter_fpr = e_b
                        epsilon = epsilon_lbf
                        if self.verbose:
                            print("NEW OPTIMAL VALUE FOUND")
                            
                        self.threshold = t

                    if self.verbose:
                        print("=============")

                if self.epsilon is not None and epsilon > self.epsilon:
                    raise ValueError("No threshold value is feasible.")
                self.epsilon = epsilon

            elif self.epsilon is not None:
                #caso ottimizzo m
                for t in candidate_threshold:
                    key_predictions = (key_scores > t)
                    nonkey_predictions = (nonkey_scores > t)

                    result = threshold_evaluate(self.epsilon,
                                                key_predictions,
                                                nonkey_predictions)
                    if result is None:
                        # epsilon_tau >= epsilon, constraint not met
                        # don't consider this value of t
                        continue

                    e_t, e_b, m_b = result

                    if m_b < self.backup_filter_size:
                        self.threshold = t
                        self.backup_filter_size = m_b
                        backup_filter_fpr = e_b


            if self.threshold is None:
                raise ValueError('No threshold value is feasible.')
        else:
            if self.backup_filter_size is None:
                raise ValueError('threshold set in LBF'
                                 ' without setting the backup filter size')
            
        all_keys = X[y]
        key_scores = self.classifier.predict_score(all_keys)
        key_predictions = (key_scores > self.threshold)

        fn_mask = ~key_predictions
        num_fn = fn_mask.sum()

        if num_fn > 0:
            #if the number of fn is very small the estimated backup_filter_fpr
            #results equal to zero and the Bloom filter raises an exception
            if backup_filter_fpr == 0.0:
                backup_filter_fpr = None

            self.backup_filter_ = \
                ClassicalBloomFilter(filter_class=self.classical_BF_class,
                                n=num_fn,  
                                epsilon=backup_filter_fpr,      
                                m=self.backup_filter_size)

            self.backup_filter_.fit(all_keys[fn_mask])
        else:
            self.backup_filter_ = None

        # TODO: is it necessary to save X and y? probably not, check this.
        # self.X_ = X
        # self.y_ = y

        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Computes predictions for a set of queries, each to be checked
        for inclusion in the Learned Bloom Filter.

        :param X: elements to classify.
        :type X: array of numerical arrays
        :return: prediction for each value in 'X'.
        :rtype: array of `bool`
        :raises: NotFittedError if the classifier is not fitted.

        NOTE: the implementation assumes that the classifier which is
        used (either pre-trained or trained in fit) refers to 1 as the
        label of keys in the Bloom filter
        """
        # TODO test that the assumption above is met for all classifiers
        # that have been tested.
        # TODO: rework after new classifier classes have been introduced

        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        scores = self.classifier.predict_score(X)

        predictions = scores > self.threshold
        if self.backup_filter_ is not None and not predictions.all():
            predictions[~predictions] = (self.backup_filter_
                                            .predict(X[~predictions]))
        return predictions

    def get_size(self):
        """Return the Learned Bloom Filter size.

        :return: size of the Learned Bloom Filter (in bits), detailed
            w.r.t. the size of the classifier and of the backup filter.
        :rtype: `dict`

        NOTE: the implementation assumes that the classifier object used to
        build the Learned Bloom Filter has a get_size method, returning the
        size of the classifier, measured in bits.
        """

        # TODO subclass all classes of the considered classifiers in order to
        # add the get_size method, also providing a flag in the constructor,
        # allowing either to compute the theoretical size or the size actually
        # occupied by the model (i.e., via json.dumps or sys.getsizeof).

        check_is_fitted(self, 'is_fitted_')

        # TODO: implement the computation of classifier size
        backup_filter_size = self.backup_filter_.get_size() \
                             if self.backup_filter_ is not None else 0
        return {'backup_filter': backup_filter_size,
                'classifier': self.classifier.get_size()}

class SLBF(BaseEstimator, BloomFilter, ClassifierMixin):
    """Implementation of the Sandwiched Learned Bloom Filter"""

    def __init__(self,
                 n=None,
                 epsilon=None,
                 m=None,
                 classifier=ScoredDecisionTreeClassifier(),
                 hyperparameters={},
                 num_candidate_thresholds=10,
                 threshold_test_size=0.7,
                 fpr_test_size=0.3,
                 model_selection_method=StratifiedKFold(n_splits=5,
                                                        shuffle=True),
                 scoring=auprc_score,
                 threshold_evaluate=threshold_evaluate,
                 classical_BF_class=ClassicalBloomFilterImpl,
                 random_state=4678913,
                 verbose=False):
        """Create an instance of :class:`SLBF`.

        :param n: number of keys, defaults to None.
        :type n: `int`
        :param epsilon: expected false positive rate, defaults to None.
        :type epsilon: `float`
        :param m: dimension in bit of the bitmap, defaults to None.
        :type m: `int`
        :param classifier: classifier to be trained, defaults to
            :class:`DecisionTreeClassifier`.
        :type classifier: :class:`sklearn.BaseEstimator`
        :param hyperparameters: grid of values for hyperparameters to be
            considered during classifier training, defaults to {}.
        :type hyperparameters: `dict`
        :param num_candidate_thresholds: number of candidate thresholds to be
            considered for mapping classifier scores onto p redictions,
            defaults to 10.
        :type num_candidate_thresholds: `int`
        :param threshold_test_size: relative validation set size used to set
            the best classifier threshold, defaults to 0.7.
        :type test_size: `float`
        :param fpr_test_size: relative test set size used to estimate
            the empirical FPR of the learnt Bloom filter, defaults
            to 0.3.
        :type fpr_test_size: `float`
        :param model_selection_method: strategy to be used for
            discovering the best hyperparameter values for the learnt
            classifier, defaults to
            `StratifiedKFold(n_splits=5, shuffle=True)`.
        :type model_selection_method:
            :class:`sklearn.model_selection.BaseShuffleSplit` or
            :class:`sklearn.model_selection.BaseCrossValidator`
        :param scoring: method to be used for scoring learnt
            classifiers, defaults to `auprc`.
        :type scoring: `str` or function
        :param threshold_evaluate: function to be used to optimize the
          classifier threshold choice (NOTE: at the current implementation
          stage there are no alternatives w.r.t. minimizing the size of the
          backup filter).
        :type threshold_evaluate: function
        :param classical_BF_class: class of the backup filter, defaults
            to :class:`ClassicalBloomFilterImpl`.
        :param random_state: random seed value, defaults to `None`,
            meaning the current seed value should be kept.
        :type random_state: `int` or `None`
        :param verbose: flag triggering verbose logging, defaults to
            `False`.
        :type verbose: `bool`
        """

        self.n = n
        self.epsilon = epsilon
        self.m = m
        self.classifier = classifier
        self.hyperparameters = hyperparameters
        self.num_candidate_thresholds = num_candidate_thresholds
        self.fpr_test_size = fpr_test_size
        self.threshold_test_size = threshold_test_size
        self.model_selection_method = model_selection_method
        self.scoring = scoring
        self.threshold_evaluate = threshold_evaluate
        self.classical_BF_class = classical_BF_class
        self.random_state = random_state
        self.verbose = verbose

    def __repr__(self):
        args = []
        if self.epsilon != None:
            args.append(f'epsilon={self.epsilon}')
        if self.classifier.__repr__() != 'ScoredDecisionTreeClassifier()':
            args.append(f'classifier={self.classifier}')
        if self.hyperparameters != {}:
            args.append(f'hyperparameters={self.hyperparameters}')
        if self.fpr_test_size != 0.3:
            args.append(f'fpr_test_size={self.fpr_test_size}')
        if (self.model_selection_method.__class__ != StratifiedKFold or 
                self.model_selection_method.n_splits != 5 or 
                self.model_selection_method.shuffle != True):
            args.append(f'model_selection_method={self.model_selection_method}')
        if callable(self.scoring):
            if self.scoring.__name__ != 'auprc_score':
                args.append(f'scoring={self.scoring.__name__}')
        else:
            score_name = args.append(f'scoring="{self.scoring}"')
        if self.random_state != 4678913:
            args.append(f'random_state={self.random_state}')
        if self.verbose != False:
            args.append(f'verbose={self.verbose}') 
        
        args = ', '.join(args)
        return f'SLBF({args})'

    def fit(self, X, y):
        """Fits the Learned Bloom Filter, training its classifier,
        setting the score threshold and building the backup filter.

        :param X: examples to be used for fitting the classifier.
        :type X: array of numerical arrays
        :param y: labels of the examples.
        :type y: array of `bool`
        :return: the fit Bloom Filter instance.
        :rtype: :class:`LBF`
        :raises: `ValueError` if X is empty, or if no threshold value is
            compliant with the false positive rate requirements.

        NOTE: If the classifier variable instance has been specified as an
        already trained classifier, `X` and `y` are considered as the dataset
        to be used to build the LBF, that is, setting the threshold for the
        in output of the classifier, and evaluating the overall empirical FPR.
        In this case, `X` and `y` are assumed to contain values not used in
        order to infer the classifier, in order to ensure a fair estimate of
        FPR. Otherwise, `X` and `y` are meant to be the examples to be used to
        train the classifier, and subsequently set the threshold and evaluate
        the empirical FPR.
        """

        # TODO: check whether or not allowing a future release in which m
        # is specified by the user, but in such case this code should be
        # placed after the classifier has been provided / trained.
        # if self.m is not None:
        #     raise NotImplementedError('LBF fixed size in constructor not yet'
        #                               ' implemented.')

        # self.n, self.epsilon, self.m = check_params_(self.n, self.epsilon,
        #                                              self.m)

        if len(X) == 0:
            raise ValueError('Empty set of keys')
        
        if self.m is None and self.epsilon is None:
            raise ValueError("At least one parameter \
                             between mand epsilon must be specified.")

        X, y = check_X_y(X, y)        

        y = check_y(y)

        if self.random_state is not None:
            self.model_selection_method.random_state = self.random_state
        
        X_pos = X[y]

        self.n = len(X_pos)

        y_pos = y[y]

        try:
            check_is_fitted(self.classifier)
            # a trained classifier was passed to the constructor
            X_neg_threshold_test = X[~y]
        except NotFittedError:
            # the classifier has to be trained
            X_neg = X[~y]

            X_neg_trainval, X_neg_threshold_test = \
                train_test_split(X_neg,
                                 test_size=self.threshold_test_size,
                                 random_state=self.random_state)
            del X_neg
            gc.collect()
            y_neg_trainval = [False] * len(X_neg_trainval)
            X_trainval = np.vstack([X_pos, X_neg_trainval])
            del X_neg_trainval
            gc.collect()

            y_trainval = np.hstack([y_pos, y_neg_trainval])
            del y_neg_trainval
            gc.collect()

            model = GridSearchCV(estimator=self.classifier,
                                 param_grid=self.hyperparameters,
                                 cv=self.model_selection_method,
                                 scoring=self.scoring,
                                 refit=True,
                                 n_jobs=-1)
            model.fit(X_trainval, y_trainval)
            del X_trainval, y_trainval
            gc.collect()
            self.classifier = model.best_estimator_

        key_scores = self.classifier.predict_score(X_pos)
        del X_pos
        gc.collect()
        nonkey_scores = self.classifier.predict_score(X_neg_threshold_test)
        del X_neg_threshold_test
        gc.collect()

        scores = np.hstack([key_scores, nonkey_scores])
        candidate_threshold = \
            np.quantile(scores,
                        np.linspace(0,
                                    1 - 1 / len(scores),
                                    self.num_candidate_thresholds,
                                    endpoint=False))

        initial_filter_size_opt = 0
        backup_filter_size_opt = m_optimal = np.inf

        # when optimizing the number of hash functions in a classical BF,
        # the probability of a false positive is alpha raised to m/n,
        # m is the size in bit of the bitmap and n is the number of keys.
        alpha = 0.5 ** math.log(2)

        # We prefer to stick to a notation analogous to that of formula
        # (2) in Mitzenmacher, A Model for Learned Bloom Filters,
        # and Optimizing by Sandwiching.
        threshold = None

        if self.m is not None:
            #fixed bit array size: optimize epsilon

            epsilon_lbf_optimal = 1
            epsilon_optimal = 1

            for t in candidate_threshold:
                key_predictions = (key_scores > t)
                n_Fn = (~key_predictions).sum() 
                Fn = (~key_predictions).sum() / len(key_predictions)
                nonkey_predictions = (nonkey_scores > t)
                Fp = nonkey_predictions.sum() / len(nonkey_predictions)

                if Fp == 1 or Fn == 1:
                    continue

                if Fp == 0:
                    #No need for initial filter, use only the lbf
                    backup_filter_size = self.m
                    initial_filter_size = 0

                elif Fn == 0:
                    # No need for backup filter in LBF, but we need inital BF
                    initial_filter_size = self.m
                    backup_filter_size = 0

                else:
                    #we need both initial and backup filter
                    b2 = Fn * math.log( \
                                Fp / ((1 - Fp) * (1/Fn - 1))) / math.log(alpha)

                    backup_filter_size = b2 * self.n
                    initial_filter_size = self.m - backup_filter_size

                    # The optimal backup filter size exceeds the given bitarray
                    # max size
                    if backup_filter_size > self.m:
                        backup_filter_size = self.m
                        initial_filter_size = 0

                #estimate the slbf FPR
                epsilon_initial_filter = alpha ** (initial_filter_size/self.n)
                epsilon_b = 0
                if n_Fn > 0:
                    epsilon_b = alpha ** (backup_filter_size / n_Fn)
                
                epsilon_lbf = Fp + (1-Fp) * epsilon_b
                epsilon_slbf = epsilon_initial_filter * (epsilon_lbf)


                if self.verbose:
                    print(f'Fp = {Fp}')
                    print(f'Fn = {Fn}')
                    print(f't={t}')
                    print(f'b1_size={initial_filter_size}')
                    print(f'b2_size={backup_filter_size}')

                if epsilon_slbf < epsilon_optimal:
                    if self.verbose:
                        print("NEW OPTIMAL VALUE FOUND")
                    epsilon_lbf_optimal = epsilon_lbf
                    epsilon_optimal = epsilon_lbf
                    optimal_b1 = initial_filter_size
                    optimal_b2 = backup_filter_size
                    threshold = t
                       
                if self.verbose:
                    print(f'=============================')

            if self.epsilon and epsilon_optimal > self.epsilon:
                raise ValueError("No threshold value is feasible.")
            self.epsilon=epsilon_optimal
            backup_filter_size = optimal_b2
            initial_filter_size = optimal_b1
        elif self.epsilon is not None:
            # fixed epsilon: optimize bit array size
            for t in candidate_threshold:
                key_predictions = (key_scores > t)
                Fn = (~key_predictions).sum() / len(key_predictions)
                nonkey_predictions = (nonkey_scores > t)
                Fp = nonkey_predictions.sum() / len(nonkey_predictions)

                if Fp == 1 or Fn == 1:
                    continue

                if Fp == 0:

                    if Fp > (1-Fn):
                        continue

                    # No need for initial filter, just build LBF with its own
                    # backup filter
                    initial_filter_size = 0

                    epsilon_b = (self.epsilon - Fp) / (1 - Fp)

                    backup_filter_size = -(Fn * self.n) * \
                                np.log(epsilon_b) / np.log(2)**2
                    
                    epsilon_lbf = Fp + (1-Fp)*epsilon_b

                elif Fn == 0:
                    backup_filter_size = 0
                    epsilon_lbf = Fp
                    # No need for backup filter in LBF, but we need inital BF
                    epsilon_initial_filter = self.epsilon/Fp
                    if epsilon_initial_filter > 1:
                        # Weird but possible case: the classifier alone has no
                        # false negatives and its false positive rate is better
                        # than the required rate for the SLBF.
                        initial_filter_size = 0
                    else:
                        initial_filter_size = -self.n * \
                                np.log(epsilon_initial_filter) / np.log(2)**2

                else:
                    if Fp < self.epsilon * (1-Fn) or Fp > (1-Fn):
                        continue

                    b2 = Fn * math.log( \
                                Fp / ((1 - Fp) * (1/Fn - 1))) / math.log(alpha)
                    epsilon_lbf = Fp + (1-Fp)* alpha ** (b2/Fn) 

                    b1 = math.log(self.epsilon / (epsilon_lbf)) \
                        / math.log(alpha)
                    
                    initial_filter_size = b1 * self.n
                    backup_filter_size = b2 * self.n                   
                m = initial_filter_size + backup_filter_size
                if m < m_optimal: 
                    m_optimal = m
                    backup_filter_size_opt = backup_filter_size
                    initial_filter_size_opt = initial_filter_size
                    epsilon_lbf_optimal = epsilon_lbf
                    threshold = t
                    if self.verbose:
                        print(f'Fp = {Fp}')
                        print(f'Fn = {Fn}')
                        print(f't={t}')
                        print(f'initial filter opt size={initial_filter_size_opt}')
                        print(f'backup filter opt size={backup_filter_size_opt}')
                        print(f'=============================')

            if m_optimal == np.inf:
                raise ValueError('No threshold value is feasible.')
            
            backup_filter_size = backup_filter_size_opt
            initial_filter_size = initial_filter_size_opt

        all_keys = X[y]
        if initial_filter_size > 0:
            self.initial_filter_ = ClassicalBloomFilter(filter_class=self.classical_BF_class,
                                                n=self.n,
                                                m=initial_filter_size)
            self.initial_filter_.fit(all_keys)
            true_mask = self.initial_filter_.predict(X)
        else:
            self.initial_filter_ = None
            true_mask = [True] * len(X)

        # TODO in this implementation, the optimal threshold is computed
        #      anew, and this is pointless, as we already know that
        #      the optimal size of the Backup filter. We can pass to the
        #      constructor both epsilon and backup filter size and check in
        #      fit of LBF: if these are provided, skip threshold analysis.
        self.lbf_ = LBF(epsilon=epsilon_lbf_optimal,
                                       classifier=self.classifier,
                                       threshold=threshold,
                                       backup_filter_size=backup_filter_size)
        self.lbf_.fit(X[true_mask], y[true_mask])

        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Computes predictions for a set of queries, each to be checked
        for inclusion in the Learned Bloom Filter.

        :param X: elements to classify.
        :type X: array of numerical arrays
        :return: prediction for each value in 'X'.
        :rtype: array of `bool`
        :raises: NotFittedError if the classifier is not fitted.

        NOTE: the implementation assumes that the classifier which is
        used (either pre-trained or trained in fit) refers to 1 as the
        label of keys in the Bloom filter
        """
        # TODO test that the assumption above is met for all classifiers
        # that have been tested.
        # TODO: rework after new classifier classes have been introduced

        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        if self.initial_filter_ is not None:
            predictions = self.initial_filter_.predict(X)
        else:
            predictions = np.array([True] * len(X))

        if len(X[predictions]) > 0:
            predictions[predictions] = self.lbf_.predict(X[predictions])

        return predictions

    def get_size(self):
        """Return the Learned Bloom Filter size.

        :return: size of the Learned Bloom Filter (in bits), detailed
            w.r.t. the size of the classifier and of the backup filter.
        :rtype: `dict`

        NOTE: the implementation assumes that the classifier object used to
        build the Learned Bloom Filter has a get_size method, returning the
        size of the classifier, measured in bits.
        """

        # TODO subclass all classes of the considered classifiers in order to
        # add the get_size method, also providing a flag in the constructor,
        # allowing either to compute the theoretical size or the size actually
        # occupied by the model (i.e., via json.dumps or sys.getsizeof).

        check_is_fitted(self, 'is_fitted_')

        initial_filter_size = self.initial_filter_.get_size() \
                             if self.initial_filter_ is not None else 0
        backup_filter_size = self.lbf_.backup_filter_.get_size() \
                             if self.lbf_.backup_filter_ is not None else 0
        return {'backup_filter': backup_filter_size,
                'initial_filter': initial_filter_size,
                'classifier': self.lbf_.classifier.get_size()}
    

class AdaBF(BaseEstimator, BloomFilter, ClassifierMixin):
    """Implementation of the Adaptive Learned Bloom Filter"""

    def __init__(self,
                 n=None,
                 epsilon=None,
                 m=None,
                 classifier=ScoredDecisionTreeClassifier(),
                 hyperparameters={},
                 threshold_test_size=0.2,
                 fpr_test_size=0.3,
                 model_selection_method=StratifiedKFold(n_splits=5,
                                                        shuffle=True),
                 scoring=auprc_score,
                 backup_filter_size=None,
                 random_state=4678913,
                 c_min = 1.6,
                 c_max = 2.5,
                 num_group_min = 8,
                 num_group_max = 12,
                 verbose=False):
        """Create an instance of :class:`AdaBF`.

        :param n: number of keys, defaults to None.
        :type n: `int`
        :param epsilon: expected false positive rate, defaults to None.
        :type epsilon: `float`
        :param m: dimension in bit of the bitmap, defaults to None.
        :type m: `int`
        :param classifier: classifier to be trained, defaults to
            :class:`DecisionTreeClassifier`.
        :type classifier: :class:`sklearn.BaseEstimator`
        :param hyperparameters: grid of values for hyperparameters to be
            considered during classifier training, defaults to {}.
        :type hyperparameters: `dict`
        :param num_candidate_thresholds: number of candidate thresholds to be
            considered for mapping classifier scores onto p redictions,
            defaults to 10.
        :type num_candidate_thresholds: `int`
        :param threshold_test_size: relative validation set size used to set
            the best classifier threshold, defaults to 0.2.
        :param fpr_test_size: relative test set size used to estimate
            the empirical FPR of the learnt Bloom filter, defaults
            to 0.3.
        :type fpr_test_size: `float`
        :param model_selection_method: strategy to be used for
            discovering the best hyperparameter values for the learnt
            classifier, defaults to
            `StratifiedKFold(n_splits=5, shuffle=True)`.
        :type model_selection_method:
            :class:`sklearn.model_selection.BaseShuffleSplit` or
            :class:`sklearn.model_selection.BaseCrossValidator`
        :param scoring: method to be used for scoring learnt
            classifiers, defaults to `auprc`.
        :type scoring: `str` or function
        :param backup_filter_size: the size of the backup filter, 
            defaults to `None`.
        :type backup_filter_size: `int`
        :param random_state: random seed value, defaults to `None`,
            meaning the current seed value should be kept.
        :type random_state: `int` or `None`
        :param c_min: min value for the c constant
            `False`.
        :type verbose: `int`
        :param c_max: min value for the c constant
            `False`.
        :type verbose: `int`
        :param num_group_min: min number of groups  
            `False`.
        :type verbose: `int`
        :param num_group_max: min number of groups  
            `False`.
        :type verbose: `int`
        :param verbose: flag triggering verbose logging, defaults to
            `False`.
        :type verbose: `bool`
        """

        self.n = n
        self.epsilon = epsilon
        self.m = m
        self.classifier = classifier
        self.hyperparameters = hyperparameters
        self.fpr_test_size = fpr_test_size
        self.threshold_test_size = threshold_test_size
        self.model_selection_method = model_selection_method
        self.scoring = scoring
        self.threshold_evaluate = threshold_evaluate
        self.backup_filter_size = backup_filter_size
        self.random_state = random_state
        self.c_min = c_min
        self.c_max = c_max
        self.num_group_min = num_group_min
        self.num_group_max = num_group_max
        self.verbose = verbose

    def __repr__(self):
        args = []
        if self.epsilon != None:
            args.append(f'epsilon={self.epsilon}')
        if self.classifier.__repr__() != 'ScoredDecisionTreeClassifier()':
            args.append(f'classifier={self.classifier}')
        if self.hyperparameters != {}:
            args.append(f'hyperparameters={self.hyperparameters}')
        if self.fpr_test_size != 0.3:
            args.append(f'fpr_test_size={self.fpr_test_size}')
        if (self.model_selection_method.__class__ != StratifiedKFold or 
            self.model_selection_method.n_splits != 5 or 
            self.model_selection_method.shuffle != True):
            args.append(f'model_selection_method={self.model_selection_method}')
        if callable(self.scoring):
            if self.scoring.__name__ != 'auprc_score':
                args.append(f'scoring={self.scoring.__name__}')
        else:
            score_name = args.append(f'scoring="{self.scoring}"')
        if self.random_state != 4678913:
            args.append(f'random_state={self.random_state}')
        if self.c_min != 1.6:
            args.append(f'c_min={self.c_min}')
        if self.c_max != 2.5:
            args.append(f'c_max={self.c_max}')
        if self.num_group_min != 8:
            args.append(f'num_group_min={self.num_group_min}')
        if self.num_group_max != 12:
            args.append(f'num_group_max={self.num_group_max}')
        if self.verbose != False:
            args.append(f'verbose={self.verbose}') 
        
        args = ', '.join(args)
        return f'AdaBF({args})'
    
    def fit(self, X, y):
        """Fits the Adaptive Learned Bloom Filter, training its classifier,
        setting the score thresholds and building the backup filter.

        :param X: examples to be used for fitting the classifier.
        :type X: array of numerical arrays
        :param y: labels of the examples.
        :type y: array of `bool`
        :return: the fit Bloom Filter instance.
        :rtype: :class:`AdaBF`
        :raises: `ValueError` if X is empty, or if no threshold value is
            compliant with the false positive rate requirements.

        NOTE: If the classifier variable instance has been specified as an
        already trained classifier, `X` and `y` are considered as the dataset
        to be used to build the LBF, that is, setting the threshold for the
        in output of the classifier, and evaluating the overall empirical FPR.
        In this case, `X` and `y` are assumed to contain values not used in
        order to infer the classifier, in order to ensure a fair estimate of
        FPR. Otherwise, `X` and `y` are meant to be the examples to be used to
        train the classifier, and subsequently set the threshold and evaluate
        the empirical FPR.
        """

        if self.m is None:
            raise ValueError('The size of the bit array must be specified')
        
        if len(X) == 0:
            raise ValueError('Empty set of keys')

        X, y = check_X_y(X, y)

        if self.random_state is not None:
            self.model_selection_method.random_state = self.random_state
        
        X_pos = X[y]
        self.n = len(X_pos)

        y_pos = y[y]

        try:
            check_is_fitted(self.classifier)
            # a trained classifier was passed to the constructor
            X_neg_threshold_test = X[~y]
        except NotFittedError:
            # the classifier has to be trained
            X_neg = X[~y]

            X_neg_trainval, X_neg_threshold_test = \
                train_test_split(X_neg,
                                 test_size=self.threshold_test_size,
                                 random_state=self.random_state)
            del X_neg
            gc.collect()
            y_neg_trainval = [False] * len(X_neg_trainval)
            X_trainval = np.vstack([X_pos, X_neg_trainval])
            del X_neg_trainval
            gc.collect()

            y_trainval = np.hstack([y_pos, y_neg_trainval])
            del y_neg_trainval
            gc.collect()

            model = GridSearchCV(estimator=self.classifier,
                                 param_grid=self.hyperparameters,
                                 cv=self.model_selection_method,
                                 scoring=self.scoring,  
                                 refit=True,
                                 n_jobs=-1)
            model.fit(X_trainval, y_trainval)
            del X_trainval, y_trainval
            gc.collect()
            self.classifier = model.best_estimator_

        c_set = np.arange(self.c_min, self.c_max+10**(-6), 0.1)

        X_neg = X[~y]
        positive_sample = X[y]
        negative_sample = X_neg
        
        nonkey_scores = np.array(self.classifier.predict_score(negative_sample))
        key_scores = np.array(self.classifier.predict_score(positive_sample))

        FP_opt = len(nonkey_scores)

        k_min = 0
        for k_max in range(self.num_group_min, self.num_group_max+1):
            for c in c_set:
                tau = sum(c ** np.arange(0, k_max - k_min + 1, 1))
                n = positive_sample.shape[0]
                bloom_filter = VarhashBloomFilter(self.m, k_max)
                thresholds = np.zeros(k_max - k_min + 1)
                thresholds[-1] = 1.1
                num_negative = sum(nonkey_scores <= thresholds[-1])
                num_piece = int(num_negative / tau) + 1
                score = nonkey_scores[nonkey_scores < thresholds[-1]]
                score = np.sort(score)
                for k in range(k_min, k_max):
                    i = k - k_min
                    score_1 = score[score < thresholds[-(i + 1)]]
                    if int(num_piece * c ** i) < len(score_1):
                        thresholds[-(i + 2)] = score_1[-int(num_piece * c ** i)]
                query = positive_sample
                score = key_scores

                my_count = 0

                for score_s, item_s in zip(score, query):
                    ix = min(np.where(score_s < thresholds)[0])
                    k = k_max - ix
                    if k > 0:
                        my_count += 1
                    bloom_filter.add(item_s, k)
                

                ML_positive = negative_sample[nonkey_scores >= thresholds[-2]]
                query_negative = negative_sample[nonkey_scores < thresholds[-2]]
                score_negative = nonkey_scores[nonkey_scores < thresholds[-2]]

                test_result = np.zeros(len(query_negative))
                ss = 0

                for score_s, item_s in zip(score_negative, query_negative):
                    ix = min(np.where(score_s < thresholds)[0])
                    k = k_max - ix
                    test_result[ss] = bloom_filter.check(item_s, k)
                    ss += 1
                FP_items = sum(test_result) + len(ML_positive)
                if self.verbose:
                    print('False positive items: %d (%f), Number of groups: %d, c = %f' %(FP_items, FP_items / len(negative_sample), k_max, round(c, 2)))

                if FP_opt > FP_items:
                    FP_opt = FP_items
                    self.backup_filter_ = bloom_filter
                    self.thresholds_ = thresholds
                    self.num_group_ = k_max

        epsilon = FP_opt / len(negative_sample)
        if self.epsilon is not None and epsilon > self.epsilon:
            raise ValueError('No threshold value is feasible.')
            
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Computes predictions for a set of queries, each to be checked
        for inclusion in the Learned Bloom Filter.

        :param X: elements to classify.
        :type X: array of numerical arrays
        :return: prediction for each value in 'X'.
        :rtype: array of `bool`
        :raises: NotFittedError if the classifier is not fitted.

        NOTE: the implementation assumes that the classifier which is
        used (either pre-trained or trained in fit) refers to 1 as the
        label of keys in the Bloom filter
        """
        # TODO test that the assumption above is met for all classifiers
        # that have been tested.
        # TODO: rework after new classifier classes have been introduced

        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        scores = np.array(self.classifier.predict_score(X))

        predictions = scores > self.thresholds_[-2]
        negative_sample = X[scores <= self.thresholds_[-2]]
        negative_scores = scores[scores <= self.thresholds_[-2]]

        ada_predictions = []
        for key, score in zip(negative_sample, negative_scores):
            ix = min(np.where(score < self.thresholds_)[0])
            # thres = thresholds[ix]
            k = self.num_group_ - ix
            ada_predictions.append(self.backup_filter_.check(key, k))

        predictions[~predictions] = np.array(ada_predictions)
        return predictions

    def get_size(self):
        """Return the Adaptive Learned Bloom Filter size.

        :return: size of the Adaptive Learned Bloom Filter (in bits), detailed
            w.r.t. the size of the classifier and of the backup filter.
        :rtype: `dict`

        NOTE: the implementation assumes that the classifier object used to
        build the Learned Bloom Filter has a get_size method, returning the
        size of the classifier, measured in bits.
        """

        check_is_fitted(self, 'is_fitted_')

        return {'backup_filter': self.m,
                'classifier': self.classifier.get_size()}
    

class PLBF(BaseEstimator, BloomFilter, ClassifierMixin):
    """Implementation of the Partitioned Learned Bloom Filter"""

    def __init__(self,
                 n=None,
                 epsilon=None,
                 m=None,
                 classifier=ScoredDecisionTreeClassifier(),
                 hyperparameters={},
                 threshold_test_size=0.2,
                 fpr_test_size=0.3,
                 model_selection_method=StratifiedKFold(n_splits=5,
                                                        shuffle=True),
                 scoring=auprc_score,
                 classical_BF_class=ClassicalBloomFilterImpl,
                 random_state=4678913,
                 num_group_min = 4,
                 num_group_max = 6,
                 N=1000,
                 verbose=False):
        """Create an instance of :class:`PLBF`.

        :param n: number of keys, defaults to None.
        :type n: `int`
        :param epsilon: expected false positive rate, defaults to None.
        :type epsilon: `float`
        :param m: dimension in bit of the bitmap, defaults to None.
        :type m: `int`
        :param classifier: classifier to be trained, defaults to
            :class:`DecisionTreeClassifier`.
        :type classifier: 
            :class:`sklearn.BaseEstimator`
        :param hyperparameters: grid of values for hyperparameters to be
            considered during classifier training, defaults to {}.
        :type hyperparameters: `dict`
        :param threshold_test_size: relative validation set size used to set
            the best classifier threshold, defaults to 0.7.
        :param fpr_test_size: relative test set size used to estimate
            the empirical FPR of the learnt Bloom filter, defaults
            to 0.3.
        :type fpr_test_size: `float`
        :param model_selection_method: strategy to be used for
            discovering the best hyperparameter values for the learnt
            classifier, defaults to
            `StratifiedKFold(n_splits=5, shuffle=True)`.
        :type model_selection_method:
            :class:`sklearn.model_selection.BaseShuffleSplit` or
            :class:`sklearn.model_selection.BaseCrossValidator`
        :param scoring: method to be used for scoring learnt
            classifiers, defaults to `auprc`.
        :type scoring: `str` or function
        :param classical_BF_class: class of the backup filter, defaults
            to :class:`ClassicalBloomFilterImpl`.
        :param random_state: random seed value, defaults to `None`,
            meaning the current seed value should be kept.
        :type random_state: `int` or `None`
        :param num_group_min: min number of groups defaults to 4
        :type num_group_min: `int`
        :param num_group_max: max number of groups, defaults to 6
        :type num_group_max: `int`
        :param N: number of segments used to discretize the classifier
            score range, defaults to 1000
        :type N: `int`
        :param verbose: flag triggering verbose logging, defaults to `False`.
        :type verbose: `bool`
        """

        self.n = n
        self.epsilon = epsilon
        self.m = m
        self.classifier = classifier
        self.hyperparameters = hyperparameters
        self.fpr_test_size = fpr_test_size
        self.threshold_test_size = threshold_test_size
        self.model_selection_method = model_selection_method
        self.scoring = scoring
        self.threshold_evaluate = threshold_evaluate
        self.classical_BF_class = classical_BF_class
        self.random_state = random_state
        self.num_group_min = num_group_min
        self.num_group_max = num_group_max
        self.verbose = verbose
        self.optim_KL = None
        self.optim_partition = None
        self.N = N

    def __repr__(self):
        args = []
        if self.epsilon != None:
            args.append(f'epsilon={self.epsilon}')
        if self.classifier.__repr__() != 'ScoredDecisionTreeClassifier()':
            args.append(f'classifier={self.classifier}')
        if self.hyperparameters != {}:
            args.append(f'hyperparameters={self.hyperparameters}')
        if self.fpr_test_size != 0.3:
            args.append(f'fpr_test_size={self.fpr_test_size}')
        if (self.model_selection_method.__class__ != StratifiedKFold or 
            self.model_selection_method.n_splits != 5 or 
            self.model_selection_method.shuffle != True):
            args.append(f'model_selection_method={self.model_selection_method}')
        if callable(self.scoring):
            if self.scoring.__name__ != 'auprc_score':
                args.append(f'scoring={self.scoring.__name__}')
        else:
            score_name = args.append(f'scoring="{self.scoring}"')
        if self.random_state != 4678913:
            args.append(f'random_state={self.random_state}')
        if self.num_group_min != 4:
            args.append(f'num_group_min={self.num_group_min}')
        if self.num_group_max != 6:
            args.append(f'num_group_max={self.num_group_max}')
        if self.N != 1000:
            args.append(f'N={self.N}')
        if self.verbose != False:
            args.append(f'verbose={self.verbose}') 
        
        args = ', '.join(args)
        return f'PLBF({args})'
    
    def fit(self, X, y):
        """Fits the Partitioned Bloom Filter, training its classifier,
        setting the score thresholds and building the backup filter.

        :param X: examples to be used for fitting the classifier.
        :type X: array of numerical arrays
        :param y: labels of the examples.
        :type y: array of `bool`
        :return: the fit Bloom Filter instance.
        :rtype: :class:`AdaBF`
        :raises: `ValueError` if X is empty, or if no threshold value is
            compliant with the false positive rate requirements.

        NOTE: If the classifier variable instance has been specified as an
        already trained classifier, `X` and `y` are considered as the dataset
        to be used to build the LBF, that is, setting the threshold for the
        in output of the classifier, and evaluating the overall empirical FPR.
        In this case, `X` and `y` are assumed to contain values not used in
        order to infer the classifier, in order to ensure a fair estimate of
        FPR. Otherwise, `X` and `y` are meant to be the examples to be used to
        train the classifier, and subsequently set the threshold and evaluate
        the empirical FPR.
        """
        
        if len(X) == 0:
            raise ValueError('Empty set of keys')
        
        if self.m is None and self.epsilon is None:
            raise ValueError("At least one parameter \
                             between m and epsilon must be specified.")

        X, y = check_X_y(X, y)

        if self.random_state is not None:
            self.model_selection_method.random_state = self.random_state
        
        X_pos = X[y]
        self.n = len(X_pos)

        y_pos = y[y]

        try:
            check_is_fitted(self.classifier)
            X_neg_threshold_test = X[~y]
        except NotFittedError:
            X_neg = X[~y]

            X_neg_trainval, X_neg_threshold_test = \
                train_test_split(X_neg,
                                 test_size=self.threshold_test_size,
                                 random_state=self.random_state)
            del X_neg
            gc.collect()
            y_neg_trainval = [False] * len(X_neg_trainval)
            X_trainval = np.vstack([X_pos, X_neg_trainval])
            del X_neg_trainval
            gc.collect()

            y_trainval = np.hstack([y_pos, y_neg_trainval])
            del y_neg_trainval
            gc.collect()

            model = GridSearchCV(estimator=self.classifier,
                                 param_grid=self.hyperparameters,
                                 cv=self.model_selection_method,
                                 scoring=self.scoring,  
                                 refit=True,
                                 n_jobs=-1)
            model.fit(X_trainval, y_trainval)
            del X_trainval, y_trainval
            gc.collect()
            self.classifier = model.best_estimator_

        X_neg = X[~y]
        X_pos = X[y]
        
        nonkey_scores = np.array(self.classifier.predict_score(X_neg_threshold_test))
        key_scores = np.array(self.classifier.predict_score(X_pos))
        FP_opt = len(nonkey_scores)

        interval = 1/self.N
        min_score = min(np.min(key_scores), np.min(nonkey_scores))
        max_score = min(np.max(key_scores), np.max(nonkey_scores))

        score_partition = np.arange(min_score-10**(-10),max_score+10**(-10)+interval,interval)

        h = [np.sum((score_low<=nonkey_scores) & (nonkey_scores<score_up)) for score_low, score_up in zip(score_partition[:-1], score_partition[1:])]
        h = np.array(h)       

        ## Merge the interval with less than 5 nonkey
        delete_ix = []
        for i in range(len(h)):
            if h[i] < 1:
                delete_ix += [i]
        score_partition = np.delete(score_partition, [i for i in delete_ix])

        h = [np.sum((score_low<=nonkey_scores) & (nonkey_scores<score_up)) for score_low, score_up in zip(score_partition[:-1], score_partition[1:])]
        h = np.array(h)
        g = [np.sum((score_low<=key_scores) & (key_scores<score_up)) for score_low, score_up in zip(score_partition[:-1], score_partition[1:])]
        g = np.array(g)

        delete_ix = []
        for i in range(len(g)):
            if g[i] < 1:
                delete_ix += [i]
        score_partition = np.delete(score_partition, [i for i in delete_ix])

        ## Find the counts in each interval
        h = [np.sum((score_low<=nonkey_scores) & (nonkey_scores<score_up)) for score_low, score_up in zip(score_partition[:-1], score_partition[1:])]
        h = np.array(h) / sum(h)
        g = [np.sum((score_low<=key_scores) & (key_scores<score_up)) for score_low, score_up in zip(score_partition[:-1], score_partition[1:])]
        g = np.array(g) / sum(g)
        
        n = len(score_partition)
        if self.optim_KL is None and self.optim_partition is None:

            optim_KL = np.zeros((n, self.num_group_max))
            optim_partition = [[0]*self.num_group_max for _ in range(n)]

            for i in range(n):
                optim_KL[i,0] = np.sum(g[:(i+1)]) * np.log2(sum(g[:(i+1)])/sum(h[:(i+1)]))
                optim_partition[i][0] = [i]

            for j in range(1,self.num_group_max):
                for m in range(j,n):
                    candidate_par = np.array([optim_KL[i][j-1]+np.sum(g[i:(m+1)])* \
                            np.log2(np.sum(g[i:(m+1)])/np.sum(h[i:(m+1)])) for i in range(j-1,m)])
                    optim_KL[m][j] = np.max(candidate_par)
                    ix = np.where(candidate_par == np.max(candidate_par))[0][0] + (j-1)
                    if j > 1:
                        optim_partition[m][j] = optim_partition[ix][j-1] + [ix] 
                    else:
                        optim_partition[m][j] = [ix]

            self.optim_KL = optim_KL
            self.optim_partition = optim_partition

        if self.m != None:

            FP_opt = len(nonkey_scores)
            
            for num_group in range(self.num_group_min, self.num_group_max+1):
                ### Determine the thresholds    
                thresholds = np.zeros(num_group + 1)
                thresholds[0] = -0.00001
                thresholds[-1] = 1.00001
                inter_thresholds_ix = self.optim_partition[-1][num_group-1]
                inter_thresholds = score_partition[inter_thresholds_ix]
                thresholds[1:-1] = inter_thresholds
                

                ### Count the keys of each group
                count_nonkey = np.zeros(num_group)
                count_key = np.zeros(num_group)

                query_group = []
                for j in range(num_group):
                    count_nonkey[j] = sum((nonkey_scores >= thresholds[j]) & (nonkey_scores < thresholds[j + 1]))
                    count_key[j] = sum((key_scores >= thresholds[j]) & (key_scores < thresholds[j + 1]))

                    query_group.append(X_pos[(key_scores >= thresholds[j]) & (key_scores < thresholds[j + 1])])


                R = np.zeros(num_group)

                alpha = 0.5 ** np.log(2)
                c = self.m / self.n + (-self.optim_KL[-1][num_group-1] / np.log2(alpha))
                
                for j in range(num_group):
                    g_j = count_key[j] / self.n
                    h_j = count_nonkey[j] / len(X_neg_threshold_test)

                    R_j = count_key[j] * (np.log2(g_j/h_j)/np.log(alpha) + c)
                    R[j] = max(1, R_j)

                #We need to fix the sizes to use all the available space
                pos_sizes_mask = R > 0
                used_bits = R[pos_sizes_mask].sum()
                relative_sizes = R[pos_sizes_mask] / used_bits
                extra_bits = self.m - used_bits

                extra_sizes = relative_sizes * extra_bits
                R[pos_sizes_mask] += extra_sizes

                for j in range(len(R)):
                    R[j] = max(1, R[j])

                backup_filters = []
                for j in range(num_group):
                    if count_key[j]==0:
                        backup_filters.append(None)
                    else:
                        backup_filters.append( \
                            ClassicalBloomFilter(filter_class=self.classical_BF_class, 
                                        n=count_key[j], 
                                        m=R[j]))
                        for item in query_group[j]:
                            backup_filters[j].add(item)

                FP_items = 0
                for score, item in zip(nonkey_scores, X_neg_threshold_test):
                    ix = min(np.where(score < thresholds)[0]) - 1
                    if backup_filters[ix] is not None:
                        FP_items += int(backup_filters[ix].check(item))

                FPR = FP_items/len(X_neg_threshold_test)

                if FP_opt > FP_items:
                    num_group_opt = num_group
                    FP_opt = FP_items
                    backup_filters_opt = backup_filters
                    thresholds_opt = thresholds
                    if self.verbose:
                        print('False positive items: {}, FPR: {} Number of groups: {}'.format(FP_items, FPR, num_group))
                        print("optimal thresholds: ", thresholds_opt)

        elif self.epsilon != None:

            m_optimal = np.inf

            for num_group in range(self.num_group_min, self.num_group_max+1):

                ### Determine the thresholds    
                thresholds = np.zeros(num_group + 1)
                thresholds[0] = -0.00001
                thresholds[-1] = 1.00001
                inter_thresholds_ix = self.optim_partition[-1][num_group-1]
                inter_thresholds = score_partition[inter_thresholds_ix]
                thresholds[1:-1] = inter_thresholds
                

                ### Count the keys of each group
                count_nonkey = np.zeros(num_group)
                count_key = np.zeros(num_group)

                query_group = []
                for j in range(num_group):
                    count_nonkey[j] = sum((nonkey_scores >= thresholds[j]) & \
                                          (nonkey_scores < thresholds[j + 1]))
                    count_key[j] = sum((key_scores >= thresholds[j]) & \
                                       (key_scores < thresholds[j + 1]))
                    query_group.append(X_pos[(key_scores >= thresholds[j]) & \
                                             (key_scores < thresholds[j + 1])])
                    g_sum = 0
                    h_sum = 0

                f = np.zeros(num_group)

                for i in range(num_group):
                    f[i] = self.epsilon * (count_key[i]/sum(count_key)) / \
                                 (count_nonkey[i]/sum(count_nonkey))

                while sum(f > 1) > 0:
                    for i in range(num_group):
                        f[i] = min(1, f[i])

                    g_sum = 0
                    h_sum = 0

                    for i in range(num_group):
                        if f[i] == 1:
                            g_sum += count_key[i] / np.sum(count_key)
                            h_sum += count_nonkey[i] / np.sum(count_nonkey)
                    
                    for i in range(num_group):
                        if f[i] < 1:

                            g_i = count_key[i]/sum(count_key)
                            h_i = count_nonkey[i]/sum(count_nonkey)
                            f[i] = g_i*(self.epsilon-h_sum) / (h_i*(1-g_sum))

                m = 0
                for i in range(num_group):
                    if f[i] < 1:
                        m += -count_key[i] * np.log(f[i]) / np.log(2)**2
                if m < m_optimal:
                    m_optimal = m
                    f_optimal = f
                    num_group_opt = num_group
                    thresholds_opt = thresholds


            count_nonkey = np.zeros(num_group_opt)
            count_key = np.zeros(num_group_opt)
            query_group = []
            for j in range(num_group_opt):
                count_nonkey[j] = sum((nonkey_scores >= thresholds[j]) & \
                                        (nonkey_scores < thresholds[j + 1]))
                count_key[j] = sum((key_scores >= thresholds[j]) & \
                                    (key_scores < thresholds[j + 1]))
                query_group.append(X_pos[(key_scores >= thresholds[j]) & \
                                            (key_scores < thresholds[j + 1])])

            backup_filters_opt = []
            for i in range(num_group_opt):
                if f_optimal[i] < 1:
                    bf = ClassicalBloomFilter(filter_class=self.classical_BF_class, 
                                        n=count_key[i], 
                                        epsilon=f[i])
                    for key in query_group[i]:
                        bf.add(key)
                    backup_filters_opt.append(bf)
                else:
                    backup_filters_opt.append(None)

        self.num_groups = num_group_opt
        self.thresholds_ = thresholds_opt
        self.backup_filters_ = backup_filters_opt
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """Computes predictions for a set of queries, each to be checked
        for inclusion in the Learned Bloom Filter.

        :param X: elements to classify.
        :type X: array of numerical arrays
        :return: prediction for each value in 'X'.
        :rtype: array of `bool`
        :raises: NotFittedError if the classifier is not fitted.

        NOTE: the implementation assumes that the classifier which is
        used (either pre-trained or trained in fit) refers to 1 as the
        label of keys in the Bloom filter
        """
        # TODO test that the assumption above is met for all classifiers
        # that have been tested.
        # TODO: rework after new classifier classes have been introduced

        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        scores = np.array(self.classifier.predict_score(X))

        counts = [0] * self.num_groups

        for j in range(self.num_groups):
            counts[j] = sum((scores >= self.thresholds_[j]) & (scores < self.thresholds_[j + 1]))

        # predictions = scores > self.__thresholds[-1]
        # negative_sample = X[scores <= self.__thresholds[-1]]
        # negative_scores = scores[scores <= self.__thresholds[-1]]

        predictions = []
        for score, item in zip(scores, X):
            ix = min(np.where(score < self.thresholds_)[0]) - 1

            if self.backup_filters_[ix] is None:
                predictions.append(True)
            else:
                predictions.append(self.backup_filters_[ix].check(item))

        return np.array(predictions)

    def get_size(self):
        """Return the Partitioned Learned Bloom Filter size.

        :return: size of the Partitioned Learned Bloom Filter (in bits), detailed
            w.r.t. the size of the classifier and of the backup filters.
        :rtype: `dict`

        NOTE: the implementation assumes that the classifier object used to
        build the Learned Bloom Filter has a get_size method, returning the
        size of the classifier, measured in bits.
        """

        check_is_fitted(self, 'is_fitted_')

        return {'backup_filters': sum([bf.m for bf in  self.backup_filters_ if bf is not None]),
                'classifier': self.classifier.get_size()}

