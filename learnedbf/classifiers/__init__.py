
import abc
import json
import numpy as np

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

def get_tree_size(tree, float_size=64, int_size=32):
    num_leaves = sum(tree.tree_.feature < 0)
    num_inner_nodes = tree.tree_.node_count - num_leaves

    # one float per leave (predicted score, fraction of pos items in node)
    # three ints per inner node (pointers to left and right subtrees,
    #                            split feature index)
    # one float per inner node (threshold)

    if hasattr(tree, "float_size"):
        float_size = tree.float_size
    if hasattr(tree, "int_size"):
        int_size = tree.int_size


    space = num_leaves * float_size \
            + num_inner_nodes * (int_size * 3 + float_size)
    return space

def _json_safe(value):
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _restore_json_array(data, dtype_spec):
    def _restore_dtype_part(value):
        if isinstance(value, list):
            return tuple(_restore_dtype_part(item) for item in value)
        return value

    def _restore_dtype(spec):
        if isinstance(spec, str):
            return np.dtype(spec)

        raw_dtype = np.dtype([
            tuple(_restore_dtype_part(item) for item in field)
            for field in spec
        ])

        named_fields = [field[0] for field in spec if field[0]]
        if len(named_fields) == len(spec):
            return raw_dtype

        return np.dtype({
            'names': named_fields,
            'formats': [raw_dtype.fields[name][0] for name in named_fields],
            'offsets': [raw_dtype.fields[name][1] for name in named_fields],
            'itemsize': raw_dtype.itemsize,
        })

    dtype = _restore_dtype(dtype_spec)
    if dtype.names is None:
        return np.array(data, dtype=dtype)

    restored = np.zeros(len(data), dtype=dtype)
    serialized_fields = [field[0] for field in dtype_spec if field[0]]

    for index, row in enumerate(data):
        for field_name, field_value in zip(serialized_fields, row):
            restored[field_name][index] = field_value

    return restored

def tree_to_json(tree):
    check_is_fitted(tree, 'tree_')

    tree_state = tree.tree_.__getstate__()
    json_tree_state = {}

    for key, value in tree_state.items():
        if isinstance(value, np.ndarray):
            json_tree_state[key] = {
                'data': _json_safe(value),
                'dtype': _json_safe(
                    value.dtype.descr if value.dtype.names else value.dtype.str
                ),
            }
        else:
            json_tree_state[key] = _json_safe(value)

    return {'params': _json_safe(tree.get_params(deep=False)),
            'classes_': _json_safe(tree.classes_),
            'n_classes_': _json_safe(tree.n_classes_),
            'n_features_in_': _json_safe(tree.n_features_in_),
            'n_outputs_': _json_safe(tree.n_outputs_),
            'max_features_': _json_safe(tree.max_features_),
            'max_features': _json_safe(tree.max_features),
            'random_state': _json_safe(tree.random_state),
            'sklearn_version': sklearn.__version__,
            'tree_state': json_tree_state,}

class ScoredClassifier:
    __metaclass__ = abc.ABCMeta

    # TODO add get_time and get_energy_consumption methods and implement
    #      them in the subclasses.

    def __init__(self, float_size=64, int_size=32, **kwargs):
        """Create an instance of :class:`ScoredClassifier`, corresponding
        to the extension of a binary classifier which outputs a classification
        confidence, called score, intended as a real number. The higher the score, the most confident is the classifier in predicting the
        positive class (that is, the class of keys stored in a Bloom filter).

        :param classifier: sklearn classifier to be scored
        :type classifier: :class:`sklearn.BaseEstimator`
        """

        self.float_size = float_size
        self.int_size = int_size

    @abc.abstractmethod
    def get_size(self, float_size=64, int_size=32):
      """Return the size in bits of the classifier.

        """
      return

    @abc.abstractmethod
    def predict_score(self, X):
       """Output the prediction score for a list of queries.

        :param X: queries to be predicted.
        :type X: array of numerical arrays.
        """
       return
    
    @abc.abstractmethod
    def to_json(self, force=False):
        """Export the classifier to a JSON representation.

        :param force: if True, the export will be forced even if the size of
                      the classifier is too large.
        :type force: bool
        """
        return
    
    def export(self, path):
        """Export the classifier to a file.

        :param path: path to the file where the classifier will be exported.
        :type path: str
        """
        with open(path, 'w') as f:
            json.dump(self.to_json(), f)
    
    @abc.abstractmethod
    def from_json(self, repr):
        """Import the classifier from a JSON representation.

        :param repr: JSON representation of the classifier.
        :type repr: dict
        """
        return
    
    def import_(self, path):
        """Import the classifier from a file.

        :param path: path to the file where the classifier will be imported
                     from.
        :type path: str
        """
        with open(path, 'r') as f:
            repr = json.load(f)

        self.from_json(repr)

class ScoredMLP(ScoredClassifier, MLPRegressor):
    """Score-based MLP classifier for a *binary* problem.
    """

    def __init__(self,
                 hidden_layer_sizes=(100,),
                 activation="relu",
                 *,
                 solver='adam',
                 alpha=0.0001,
                 batch_size='auto',
                 learning_rate="constant",
                 learning_rate_init=0.001,
                 power_t=0.5,
                 max_iter=200,
                 shuffle=True,
                 random_state=None,
                 tol=1e-4,
                 verbose=False,
                 warm_start=False,
                 momentum=0.9,
                 nesterovs_momentum=True,
                 early_stopping=False,
                 validation_fraction=0.1,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 n_iter_no_change=10,
                 max_fun=15000,
                 float_size=64,
                 int_size=32):
      
        MLPRegressor.__init__(self,
                              hidden_layer_sizes=hidden_layer_sizes, 
                              activation=activation,
                              solver=solver,
                              alpha=alpha,
                              batch_size=batch_size, 
                              learning_rate=learning_rate,
                              learning_rate_init=learning_rate_init,
                              power_t=power_t,
                              max_iter=max_iter,
                              shuffle=shuffle,
                              random_state=random_state,
                              tol=tol,
                              verbose=verbose,
                              warm_start=warm_start,
                              momentum=momentum,
                              nesterovs_momentum=nesterovs_momentum, 
                              early_stopping=early_stopping,
                              validation_fraction=validation_fraction,
                              beta_1=beta_1,
                              beta_2=beta_2,
                              epsilon=epsilon,
                              n_iter_no_change=n_iter_no_change, max_fun=max_fun)
        
        ScoredClassifier.__init__(self,
                                     float_size=float_size,
                                     int_size=int_size)
        
    def fit(self, X, y):
        #super(MLPRegressor, self).fit(X, y)
        MLPRegressor.fit(self, X, y)
        self.out_activation_ = 'logistic'
        return self

    def predict_score(self, X):
        check_is_fitted(self, 'n_features_in_')
        return self.predict(X)
    
    def get_size(self):
        check_is_fitted(self, 'n_features_in_')
        hidden_layer_sizes = np.array(self.hidden_layer_sizes)

        first = np.insert(hidden_layer_sizes,
                          0, self.n_features_in_)
        # we add 1 to take into account biases
        first += np.ones(len(first)).astype(int)
        second = np.append(hidden_layer_sizes, 1)
        num_connections = np.dot(first, second)
        return num_connections * self.float_size


class ScoredLinearSVC(ScoredClassifier, LinearSVC):
    """Score-based linear SV classifier for a *binary* problem.
    """
    
        
    def __init__(self,
                 penalty='l2',
                 loss='squared_hinge',
                 *,
                 dual=True,
                 tol=1e-4,
                 C=1.0,
                 multi_class='ovr',
                 fit_intercept=True,
                 intercept_scaling=1,
                 class_weight=None,
                 verbose=0,
                 random_state=None,
                 max_iter=1000,
                 float_size=64,
                 int_size=32):
        
        LinearSVC.__init__(self,
                           penalty=penalty,
                           loss=loss,
                           dual=dual,
                           tol=tol,
                           C=C,
                           multi_class=multi_class,
                           fit_intercept=fit_intercept,
                           intercept_scaling=intercept_scaling,
                           class_weight=class_weight,
                           verbose=verbose,
                           random_state=random_state,
                           max_iter=max_iter)
        
        ScoredClassifier.__init__(self,
                                     float_size=float_size,
                                     int_size=int_size)

    def predict_score(self, X):
        check_is_fitted(self, 'n_features_in_')

        if 1 not in self.classes_ and True not in self.classes_:
            raise ValueError('LinearSVC not trained using'
                            'either 1 or True as positive label' )

        # Note that LinearSVC decides which is the positive class
        # depending on the order of examples passed to fit
        # in any case, the first value in the .classes_ attribute corresponds
        # to the chosen positive class.
        # Therefore, if we know that the positive class is False / 0, the
        # decision function value should be changed in sign.
        
        decision = self.decision_function(X)
        if self.classes_[0] is False or self.classes_[0] == 0:
            decision = - decision

        # To avoid numerical issues with the exponential function, we compute the sigmoid function in a piecewise way, as suggested in https://stackoverflow.com/questions/51976461/logistic-function-overflow-in-python
        out = np.empty_like(decision, dtype=float)
        pos = decision >= 0
        out[pos]  = np.exp(-decision[pos]) / (1.0 + np.exp(-decision[pos]))
        out[~pos] = 1.0 / (1.0 + np.exp(decision[~pos]))

        #return 1 / (1 + np.exp(decision))
        return out

    def get_size(self):
        check_is_fitted(self, 'n_features_in_')
        return (1 + self.n_features_in_) * self.float_size
    
    def to_json(self, force=False):
        check_is_fitted(self, 'n_features_in_')
        if not force and self.get_size() > 1e6:
            raise ValueError('The size of the classifier is too large to be exported. Use force=True to override this check.')
        return {'params': _json_safe(self.get_params(deep=False)),
                'coef_': self.coef_.tolist(),
                'intercept_': self.intercept_.tolist(),
                'classes_': self.classes_.tolist(),
                'float_size': self.float_size,
                'int_size': self.int_size,
                'n_features_in_': self.n_features_in_,}
    def from_json(self, repr):
        self.set_params(**repr['params'])
        self.coef_ = np.array(repr['coef_'])
        self.intercept_ = np.array(repr['intercept_'])
        self.classes_ = np.array(repr['classes_'])
        self.float_size = repr['float_size']
        self.int_size = repr['int_size']
        self.n_features_in_ = repr['n_features_in_']
        return self

    def __eq__(self, other):
        if self is other:
            return True

        if not isinstance(other, ScoredLinearSVC):
            return NotImplemented

        if self.get_params(deep=False) != other.get_params(deep=False):
            return False

        self_fitted = (hasattr(self, "coef_")
                       and hasattr(self, "intercept_")
                       and hasattr(self, "classes_"))
        other_fitted = (hasattr(other, "coef_")
                        and hasattr(other, "intercept_")
                        and hasattr(other, "classes_"))

        if self_fitted != other_fitted:
            return False

        if not self_fitted:
            return True

        return (np.array_equal(self.coef_, other.coef_)
                and np.array_equal(self.intercept_, other.intercept_)
                and np.array_equal(self.classes_, other.classes_))


class ScoredDecisionTreeClassifier(ScoredClassifier, DecisionTreeClassifier):
    """Score-based Decision Tree classifier for a *binary* problem.
    """

    def __init__(self, *,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 class_weight=None,
                 ccp_alpha=0.0,
                 float_size=64,
                 int_size=32):
        
        DecisionTreeClassifier.__init__(self,
                        criterion=criterion,
                        splitter=splitter,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                        max_features=max_features,
                        random_state=random_state,
                        max_leaf_nodes=max_leaf_nodes,
                        min_impurity_decrease=min_impurity_decrease,
                        class_weight=class_weight,
                        ccp_alpha=ccp_alpha)
        
        ScoredClassifier.__init__(self, float_size, int_size)
        
    def predict_score(self, X):
        check_is_fitted(self, 'classes_')
        score_dict = [{c: p for c, p in zip(self.classes_, x)}
                      for x in self.predict_proba(X)]
        
        if 1 not in self.classes_ and True not in self.classes_:
            raise ValueError('DecisionTreeClassifier not trained using'
                            'either 1 or True as positive label' )

        pos_key = 1 if 1 in self.classes_ else True
        
        return [d[pos_key] for d in score_dict]

    def get_size(self):
        check_is_fitted(self, 'tree_')
        return get_tree_size(self)

    def __eq__(self, other):
        if self is other:
            return True

        if not isinstance(other, ScoredDecisionTreeClassifier):
            return NotImplemented

        if self.get_params(deep=False) != other.get_params(deep=False):
            return False

        self_fitted = hasattr(self, "tree_") and hasattr(self, "classes_")
        other_fitted = hasattr(other, "tree_") and hasattr(other, "classes_")

        if self_fitted != other_fitted:
            return False

        if not self_fitted:
            return True

        if not np.array_equal(self.classes_, other.classes_):
            return False

        if getattr(self, "n_features_in_", None) != \
                              getattr(other, "n_features_in_", None):
            return False

        self_tree = self.tree_.__getstate__()
        other_tree = other.tree_.__getstate__()

        if self_tree.keys() != other_tree.keys():
            return False

        for key in self_tree:
            left = self_tree[key]
            right = other_tree[key]

            if isinstance(left, np.ndarray):
                if not np.array_equal(left, right):
                    return False
            else:
                if left != right:
                    return False

        return True

    def to_json(self, force=False):
        check_is_fitted(self, 'tree_')
        if not force and self.get_size() > 1e6:
            raise ValueError(
                'The size of the classifier is too large to be exported. '
                'Use force=True to override this check.'
            )

        tree_state = self.tree_.__getstate__()
        json_tree_state = {}

        for key, value in tree_state.items():
            if isinstance(value, np.ndarray):
                json_tree_state[key] = {
                    'data': _json_safe(value),
                    'dtype': _json_safe(
                        value.dtype.descr if value.dtype.names else value.dtype.str
                    ),
                }
            else:
                json_tree_state[key] = _json_safe(value)

        return {'params': _json_safe(self.get_params(deep=False)),
                'classes_': _json_safe(self.classes_),
                'n_classes_': _json_safe(self.n_classes_),
                'n_features_in_': _json_safe(self.n_features_in_),
                'n_outputs_': _json_safe(self.n_outputs_),
                'max_features_': _json_safe(self.max_features_),
                'float_size': _json_safe(self.float_size),
                'int_size': _json_safe(self.int_size),
                'sklearn_version': sklearn.__version__,
                'tree_state': json_tree_state,}

    def from_json(self, repr):
        from sklearn.tree import _tree

        saved_version = repr.get('sklearn_version')
        current_version = sklearn.__version__
        if saved_version is not None and saved_version != current_version:
            raise ValueError(
                f'Cannot import decision tree serialized with scikit-learn '
                f'{saved_version} using scikit-learn {current_version}.'
            )

        self.set_params(**repr['params'])

        self.classes_ = np.array(repr['classes_'])

        n_classes = repr['n_classes_']
        if isinstance(n_classes, list):
            self.n_classes_ = np.array(n_classes, dtype=np.intp)
            tree_n_classes = self.n_classes_.astype(np.intp, copy=False)
        else:
            self.n_classes_ = int(n_classes)
            tree_n_classes = np.array([self.n_classes_], dtype=np.intp)

        self.n_features_in_ = int(repr['n_features_in_'])
        self.n_outputs_ = int(repr['n_outputs_'])
        self.max_features_ = repr['max_features_']
        self.float_size = int(repr['float_size'])
        self.int_size = int(repr['int_size'])

        tree_state = {}
        for key, value in repr['tree_state'].items():
            if isinstance(value, dict) and 'data' in value and 'dtype' in value:
                tree_state[key] = _restore_json_array(
                    value['data'],
                    value['dtype'],
                )
            else:
                tree_state[key] = value

        self.tree_ = _tree.Tree(
            self.n_features_in_,
            tree_n_classes,
            self.n_outputs_,
        ) # note that this will flag the tree as fitted
        self.tree_.__setstate__(tree_state)

        return self
    

class ScoredRandomForestClassifier(ScoredClassifier, RandomForestClassifier):
    """Score-based Random Forest classifier for a *binary* problem.
    """
    def __init__(self,
                 n_estimators=100,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="sqrt",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None,
                 float_size=64,
                 int_size=32):
        
        RandomForestClassifier.__init__(self,
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                class_weight=class_weight,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples)
        
        ScoredClassifier.__init__(self, float_size, int_size)
        

    def predict_score(self, X):
        check_is_fitted(self, 'classes_')
        score_dict = [{c: p for c, p in zip(self.classes_, x)}
                      for x in self.predict_proba(X)]
        
        if 1 not in self.classes_ and True not in self.classes_:
            raise ValueError('RandomForestClassifier not trained using'
                             'either 1 or True as positive label' )

        pos_key = 1 if 1 in self.classes_ else True
        
        return [d[pos_key] for d in score_dict]
    
    def get_size(self):
        check_is_fitted(self, 'estimators_')
        return sum([get_tree_size(t, float_size=self.float_size, int_size=self.int_size) for t in self.estimators_])

    def to_json(self, force=False):
        check_is_fitted(self, 'estimators_')
        if not force and self.get_size() > 1e6:
            raise ValueError(
                'The size of the classifier is too large to be exported. '
                'Use force=True to override this check.'
            )
        
        return {
            #'n_estimators': _json_safe(self.n_estimators),
            'params': _json_safe(self.get_params(deep=False)),
            'estimators_': [tree_to_json(e) for e in self.estimators_],
            'classes_': _json_safe(self.classes_),
            'n_classes_': _json_safe(self.n_classes_),
            'n_features_in_': _json_safe(self.n_features_in_),
            'n_outputs_': _json_safe(self.n_outputs_),
            #'float_size': _json_safe(self.float_size),
            #'int_size': _json_safe(self.int_size),
            'sklearn_version': sklearn.__version__,
        }

    def from_json(self, repr):
        from sklearn.tree import _tree

        saved_version = repr.get('sklearn_version')
        current_version = sklearn.__version__
        if saved_version is not None and saved_version != current_version:
            raise ValueError(
                f'Cannot import decision tree serialized with scikit-learn '
                f'{saved_version} using scikit-learn {current_version}.'
            )

        self.set_params(**repr['params'])

        self.classes_ = np.array(repr['classes_'])

        n_classes = repr['n_classes_']
        if isinstance(n_classes, list):
            self.n_classes_ = np.array(n_classes, dtype=np.intp)
            tree_n_classes = self.n_classes_.astype(np.intp, copy=False)
        else:
            self.n_classes_ = int(n_classes)
            tree_n_classes = np.array([self.n_classes_], dtype=np.intp)

        self.n_features_in_ = int(repr['n_features_in_'])
        self.n_outputs_ = int(repr['n_outputs_'])
        
        self.max_features = repr['params']['max_features']
        #self.float_size = int(repr['float_size'])
        #self.int_size = int(repr['int_size'])

        self.estimators_ = []
        for estimator_repr in repr['estimators_']:
            tree_state = {}
            for key, value in estimator_repr['tree_state'].items():
                if isinstance(value, dict) and 'data' in value and 'dtype' in value:
                    tree_state[key] = _restore_json_array(
                        value['data'],
                        value['dtype'],
                    )
                else:
                    tree_state[key] = value

            t = DecisionTreeClassifier()
            t.tree_ = _tree.Tree(
               self.n_features_in_,
               tree_n_classes,
               self.n_outputs_,
            ) # note that this will flag the tree as fitted
            t.classes_ = np.array(estimator_repr['classes_'])
            t.n_classes_ = estimator_repr['n_classes_']
            t.n_features_in_ = int(estimator_repr['n_features_in_'])
            t.n_outputs_ = int(estimator_repr['n_outputs_'])
            t.max_features_ = estimator_repr['max_features_']
            t.max_features = estimator_repr['max_features']
            t.random_state = estimator_repr['random_state']
            t.tree_.__setstate__(tree_state)

            self.estimators_.append(t)

        return self