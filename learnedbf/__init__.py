import abc
import gc
import logging
import math
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import auc, precision_recall_curve, make_scorer, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV, \
                                    train_test_split
from sklearn.utils.validation import NotFittedError, check_X_y, check_array, \
                                     check_is_fitted
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

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
            print(f'lenght of x_neg_threshold_test is {len(X_neg_threshold_test)}')
            print('-----------------------')
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
            print(f'lenght of x_neg_threshold_test is {len(X_neg_threshold_test)}')
            nonkey_scores = self.classifier.predict_score(X_neg_threshold_test)
            del X_neg_threshold_test
            gc.collect()

            scores = np.hstack([key_scores, nonkey_scores])

            unique_scores = np.unique(scores)
            n_unique = len(unique_scores)
            if n_unique <= self.num_candidate_thresholds:
                self.num_candidate_thresholds = n_unique - 1
                candidate_threshold = np.sort(unique_scores)[:-1]
            else:
                candidate_threshold = \
                    np.quantile(scores,
                                np.linspace(0,
                                            1 - 1 / len(scores),
                                            self.num_candidate_thresholds))

            self.backup_filter_size = np.inf
            self.threshold = None

            if self.m is not None:
                print('non devo entrare qui')
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
                print('invece DEVO entrare qui')
                #caso ottimizzo m
                print(f'candidate threshold = {candidate_threshold}')
                print(f'nonkey-scores lenght is {len(nonkey_scores)}')
                for t in candidate_threshold:
                    key_predictions = (key_scores > t)
                    nonkey_predictions = (nonkey_scores > t)

                    nonkey_predictions_temp = (nonkey_scores >= t)
                    epsilon_tau = nonkey_predictions_temp.sum() / len(nonkey_predictions_temp)
                    if epsilon_tau == 0:
                        print('\t\tepsilon tau is zero!')

                    result = threshold_evaluate(self.epsilon,
                                                key_predictions,
                                                nonkey_predictions)
                    if result is None:
                        # epsilon_tau >= epsilon, constraint not met
                        # don't consider this value of t
                        print('skipping!')
                        continue

                    e_t, e_b, m_b = result

                    print(f'tau={t:.2f}, e_t={e_t:.2f}, e_b={e_b:.2f}, m_b={m_b:.2f}')
                    print(f'    epsilon for filter: {e_t + (1 - e_t) * e_b}')

                    if m_b == 0 and e_t <= self.epsilon:
                        self.threshold = t
                        self.backup_filter_ = None
                        break


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

        y = check_y(y)

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

class FLBF(BaseEstimator, BloomFilter, ClassifierMixin):
    """Implementation of the Fully Learned Bloom Filter"""

    def __init__(self,
                 n = None,
                 epsilon = None,
                 t = None,
                 fpr_test_size = 0.3,
                 min_hidden_size = 1,
                 max_hidden_size = 20,
                 sigma = 0.3,
                 C_n_min = 0.1,
                 C_n_max = 20,
                 random_state = 4678913,
                 verbose = False):
        """Create an instance of :class:`FLF`.

        :param n: number of keys, defaults to None.
        :type n: `int`

        :param t: maximum number of classifiers composing the filter, defaults to None.
        :type t: `int`

        :param epsilon: expected false positive rate, defaults to None.
        :type epsilon: `float`

        :param fpr_test_size: relative test set size used to estimate
            the empirical FPR of the learnt Bloom filter, defaults
            to 0.3.
        :type fpr_test_size: `float`

        :param min_hidden_size: minimum number of neurons for each mlp's hidden layers,
            defaults to 1.
        :type min_hidden_size: `int`

        :param max_hidden_size: maximum number of neurons for each mlp's hidden layers,
            defaults to 20.
        :type max_hidden_size: `int`

        :param sigma: error threshold for each classifier's FPR, defaults to 0.3.
        :type sigma: `float`

        :param C_n_min: minimum negative class weight for each classifier, defaults to 0.1.
        :type C_n_min: `float`

        :param C_n_max: maximum negative class weight for each classifier, defaults to 20.
        :type C_n_max: `float`

        :param random_state: random seed value, defaults to `None`,
            meaning the current seed value should be kept.
        :type random_state: `int` or `None`

        :param verbose: flag triggering verbose logging, defaults to `False`.
        :type verbose: `bool`
        """

        self.n = n
        self.epsilon = epsilon
        self.t = t
        self.fpr_test_size = fpr_test_size
        self.min_hidden_size = min_hidden_size
        self.max_hidden_size = max_hidden_size
        self.sigma = sigma
        self.C_n_min = C_n_min
        self.C_n_max = C_n_max
        self.random_state = random_state
        self.verbose = verbose

    def __repr__(self):
        args = []
        if self.epsilon != None:
            args.append(f'epsilon={self.epsilon}')
        if self.t != None:
            args.append(f't={self.t}')
        if self.fpr_test_size != 0.3:
            args.append(f'fpr_test_size={self.fpr_test_size}')
        if self.random_state != 4678913:
            args.append(f'random_state={self.random_state}')
        if self.verbose != False:
            args.append(f'verbose={self.verbose}') 
        
        args = ', '.join(args)
        return f'FLBF({args})'
    
    def fit(self, X, y):
        """Fits the Fully Learned Bloom Filter.

        :param X: examples to be used for fitting the classifier.
        :type X: array of numerical arrays
        :param y: labels of the examples.
        :type y: array of `bool`
        :return: the fit Bloom Filter instance.
        :rtype: :class:`FLBF`
        :raises: `ValueError` if X is empty.
        """

        if self.epsilon is None:
            raise ValueError("epsilon must be specified.")
            
        if len(X) == 0:
            raise ValueError('Empty set of keys')

        X, y = check_X_y(X, y)
        
        X_pos = X[y]

        self.n = len(X_pos)

        neg_indices = y == False
        neg_X, neg_y = X[neg_indices], y[neg_indices]
        X_train, X_test, y_train, y_test = train_test_split(
            neg_X, neg_y, test_size=self.fpr_test_size, random_state=self.random_state)
        X_train = np.concatenate([X_pos, X_train], axis=0)
        y_train = np.concatenate([y[y], y_train], axis=0)

        s = math.ceil(self.n * math.log((1/self.epsilon), 2) / math.log(2))
        z = 1 - ((1 - self.epsilon) ** (1/self.t))
        space_left = s
        self.chain = []
        self.sizes = []
        fprs = []

        i = 0
        while i < self.t:

            positive_count = np.count_nonzero(y_train == True)
            negative_count = np.count_nonzero(y_train == False)
            
            if self.verbose:
                print(f"{positive_count} keys left ~ {negative_count} negatives.")

            if i == self.t-1:
                # Last iteration

                if len(self.chain) == 0:
                    if self.verbose:
                        print("No mlp's were added, aborting.")
                    # TODO da definire il comportamento in qs caso
                    break

                # Compute z based on all the previous classifiers performances
                z = 1 - ((1-self.epsilon) / math.prod([1 - fpr for fpr in fprs]))

                max_leaves = (space_left + 128) // 136
                if self.verbose:
                    print(f"Space left for tree is {space_left} => max leaves = {max_leaves}.")
                model, fpr = self._train_smallest_tree(
                    X_train, y_train, X_test, y_test, z, max_leaves, 
                    self.random_state, verbose=self.verbose)
                size = self._tree_size(model)

                use_bf = False

                if model is None:
                    # Largest possible tree cannot reach z, use a BF.
                    if self.verbose:
                        print("Largest tree cannot reach z. BF will be used instead.")
                    use_bf = True
                else:
                    # Found a valid tree
                    # See if BF is more efficient; if so, replace tree
                    equivalent_bf_size = math.ceil(positive_count * math.log((1/fpr), 2) / math.log(2))
                    ratio = size / equivalent_bf_size
                    if self.verbose:
                        print(f"Space ratio: {ratio}.")
                    if ratio > 1:
                        if self.verbose:
                            print(f"Replacing the DT with a BF. Space ratio was > 1.")
                        use_bf = True

                if use_bf:
                    model = ClassicalBloomFilter(n=positive_count, epsilon=z)
                    model.fit(X_train[y_train])
                    size = model.get_size()
                    fpr = model.estimate_FPR(X_test)

                self.chain.append(model)
                self.sizes.append(size)
                fprs.append(fpr)
                if self.verbose:
                    print(f"Added a {'BF' if use_bf else 'DT'} with size {size} \
                        and FPR {fpr}; z was {z}.")
                break
            else:
                # Non-last iterations

                # Adjust z according to the previous models' FPR's (if there are any)
                if len(self.chain) > 0:
                    z = 1 - ((1-self.epsilon) / math.prod([1 - fpr for fpr in fprs])) \
                        **(1 / (self.t - len(self.chain)))
                    if z < 0:
                        # TODO this might be useless
                        if self.verbose:
                            print(f"Cannot reach desired FPR, adjusted z was {z}.")
                        break

                model, fpr, size = self._train_best_mlp(
                    X_train, y_train, X_test, y_test, z, self.random_state, 
                    self.max_hidden_size, self.min_hidden_size, self.sigma,
                    self.C_n_min, self.C_n_max, verbose=self.verbose)
                
                # If the mlp is not valid, skip to the last iteration
                if model is None:
                    if self.verbose:
                        print(f"Skipping to final classifier after {i} added mlp's. \
                            Last one had intolerable FPR.")
                    i = self.t-1
                    continue
                
                y_hat_train = model.predict(X_train)

                # Compute TPs, FNs
                cm = confusion_matrix(y_train, y_hat_train, labels=[False, True])
                tp = cm[1][1]
                fn = cm[1][0]

                if tp == 0:
                    if self.verbose:
                        print(f"Skipping to final classifier after {i} added mlp's. \
                              No true positives.")
                    i = self.t-1
                    continue

                # Calculate space ratio between current classifier and equivalent bloom filter
                equivalent_bf_size = math.ceil(tp * math.log((1/fpr), 2) / math.log(2))
                ratio = size / equivalent_bf_size
                if self.verbose:
                    print(f"Space ratio: {ratio}")
                if ratio > 1:
                    if self.verbose:
                        print(f"Skipping to final classifier after {i} added mlp's. \
                                Space ratio was > 1.")
                    i = self.t-1
                    continue

                # Current mlp is valid, we can append it to the chain
                self.chain.append(model)
                self.sizes.append(size)
                fprs.append(fpr)
                space_left -= size
                if self.verbose:
                    print(f"Added a mlp with size {size} and FPR {fpr}; z was {z}.")

                if fn == 0:
                    if self.verbose:
                        print("No false negatives, end of training.")
                    break

                # Keep only samples classified as negative
                train_negatives_idx = y_hat_train == False
                X_train, y_train = X_train[train_negatives_idx], y_train[train_negatives_idx]
                y_hat_test = model.predict(X_test)
                test_negatives_idx = y_hat_test == False
                X_test, y_test = X_test[test_negatives_idx], y_test[test_negatives_idx]

                i += 1
        
        self.is_fitted_ = True
        return self

    def _mlp_size(self, in_neurons, hidden_layers, out_neurons=1, bits_per_param=32):
        """
        Returns the size of the network [bits].
        """
        
        layers = [in_neurons] + hidden_layers + [out_neurons]
        total_weights = sum(layers[i] * layers[i + 1] for i in range(len(layers) - 1))
        total_biases = sum(hidden_layers) + out_neurons

        return (total_weights + total_biases) * bits_per_param

    def _train_best_mlp(self, X_train, y_train, X_test, y_test, z, random_state, 
                     max_hidden_size, min_hidden_size, sigma,
                     C_n_min, C_n_max, verbose=False):
        """
        Trains a mlp with FPR close to z by doing a binary search on C_n 
        and a linear search on hidden layer size.
        Returns None if the best mlp found does not have a tolerable FPR.
        """

        hidden_layers_size = min_hidden_size
        
        while True:
            if verbose:
                print(f"Training a MLP with hidden layers size of {hidden_layers_size}")
            
            C_n_inf, C_n_sup = C_n_min, C_n_max
            
            while True:
                curr_C_n = (C_n_sup + C_n_inf) / 2
            
                # Train a mlp with negative weight of curr_C_n
                sample_weights = [curr_C_n if not i else 1 for i in y_train]
                mlp = MLPClassifier(
                    hidden_layer_sizes=[hidden_layers_size]*2, 
                    max_iter=2000, random_state=random_state)
                mlp.fit(X_train, y_train, sample_weights)
                
                # Evaluate FPR on test set and halve C_n range accordingly
                FPR = self._false_positive_rate(y_test, mlp.predict(X_test))
                delta = z - FPR
                
                if delta >= 0:
                    C_n_sup = curr_C_n
                elif delta < 0:
                    C_n_inf = curr_C_n
                    
                if C_n_sup - C_n_inf <= 1e-6:
                    # End of binary search on C_n
                    if verbose:
                        print(f"Best mlp has FPR = {FPR}, z = {z} => delta {delta}")
                    break
            
            # See whether to increase or decrease network size
            if FPR < z * (1 - sigma) or FPR > z * (1 + sigma):
                if verbose:
                    print(f"FPR is not in range [{z * (1 - sigma)}, {z * (1 + sigma)}]")
                # Need a bigger network
                if hidden_layers_size == max_hidden_size:
                    # Already at maximum size
                    return None, 0, 0
                
                hidden_layers_size += 1
                    
            else:
                # Found a valid network
                return mlp, FPR, self._mlp_size(X_train.shape[1], [hidden_layers_size]*2)

    def _train_best_tree(self, X, y, class_weight_list, seed, max_leaf_nodes=None, verbose=False):
        best_FNR = 1
        # Loop over the class weights and train a decision tree for each
        for weight in class_weight_list:
            # Initialize the DecisionTreeClassifier with the current class weight
            dtc = DecisionTreeClassifier(
                class_weight=weight, max_leaf_nodes=max_leaf_nodes,random_state=seed)
            
            dtc.fit(X, y)
            y_pred = dtc.predict(X)
            cm = confusion_matrix(y, y_pred, labels=[False, True])
            fn = cm[1][0]
            tp = cm[1][1]
            fnr = fn / (fn + tp)
            
            if verbose:
                print("\t class_weight:", weight)
                print(f"False Negative Rate: {fnr}")

            if fnr < best_FNR:
                best_FNR = fnr
                best_estimator = dtc
                best_weight = weight
            if best_FNR == 0:
                break    
        
        return best_estimator, best_weight, best_FNR

    def _false_positive_rate(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[False, True])
        tn, fp, _, _ = cm.ravel()
        return fp / (fp + tn)

    def _tree_size(self, tree):
        if tree is None:
            return 0
        n_leaves = tree.tree_.n_leaves
        n_inner = tree.tree_.node_count-n_leaves
        space_in_bit= 8*n_leaves + 128*n_inner
        return space_in_bit

    def _train_smallest_tree(
            self, X_train, y_train, X_test, y_test, z, 
            max_leaves, random_state, verbose=False):
        """
        Trains the smallest tree that has FN=0, FPR<=z and max_leaves leaves at most.
        If there's no such tree, returns None.
        """
        
        class_weights = [{False: 1, True: w} for w in np.arange(1, 50.1, 0.1)]

        # See if biggest tree can reach FPR<=z. If not, rule out DT entirely.
        model, _, fnr = self._train_best_tree(X_train, y_train, class_weights, 
                                random_state, max_leaf_nodes=max_leaves, verbose=verbose)
        y_hat_test = model.predict(X_test)
        fpr = self._false_positive_rate(y_test, y_hat_test)
        if fpr > z or fnr > 0:
            if verbose:
                print(f"Ruled out DT immediately: largest tree is not sufficient.")
            return None, 0

        sup = max_leaves
        inf = 2
        smallest = [model, fpr]

        while inf <= sup:
            leaves = (sup + inf) // 2
            if verbose:
                print(f"Training tree with max leaves = {leaves}.")
            model, C_p, fnr = self._train_best_tree(
                X_train, y_train, class_weights, random_state, 
                max_leaf_nodes=leaves, verbose=verbose)
            
            y_hat_test = model.predict(X_test)
            fpr = self._false_positive_rate(y_test, y_hat_test)
            
            if verbose:
                print(f"FPR = {fpr}, z = {z}.")

            if fpr > z or fnr > 0:
                # Need to increase number of leaves
                # First, check if BF with FPR=z is already more space-efficient
                y_hat_train = model.predict(X_train)
                cm = confusion_matrix(y_hat_train, y_train, labels=[False, True])
                tp = cm[1][1]
                equivalent_bf_size = math.ceil(tp * math.log((1/z), 2) / math.log(2))
                size = self._tree_size(model)
                ratio = size / equivalent_bf_size
                if ratio > 1:
                    print(f"BF with FPR=z is already more efficient (ratio={ratio}).")
                    return smallest
                
                inf = leaves + 1
            else:
                # Found a valid DT
                smallest = model, fpr
                # Can decrease number of leaves
                sup = leaves - 1

        return smallest

    def predict(self, X):
        """Computes predictions for a set of queries, each to be checked
        for inclusion in the Learned Bloom Filter.

        :param X: elements to classify.
        :type X: array of numerical arrays
        :return: prediction for each value in 'X'.
        :rtype: array of `bool`
        :raises: NotFittedError if the classifier is not fitted.
        """

        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        predictions = np.zeros(len(X))
        for model in self.chain:
            y_hat = model.predict(X)
            predictions[y_hat] = 1

        return predictions

    def get_size(self):
        """Returns the size of the Fully Learned Bloom Filter.

        :return: size in bits of each classifier in the chain.
        :rtype: `tuple`
        :raises: NotFittedError if the classifier is not fitted.
        """

        check_is_fitted(self, 'is_fitted_')
        return tuple(self.sizes)