
import abc
import numpy as np

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

    @abc.abstractclassmethod
    def get_size(self, float_size=64, int_size=32):
      """Return the size in bits of the classifier.

        """
      return

    @abc.abstractclassmethod
    def predict_score(self, X):
       """Output the prediction score for a list of queries.

        :param X: queries to be predicted.
        :type X: array of numerical arrays.
        """
       return
    

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
        super(MLPRegressor, self).fit(X, y)
        self.out_activation_ = 'logistic'

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

        return 1 / (1 + np.exp(decision))
    
    def get_size(self):
        check_is_fitted(self, 'n_features_in_')
        return (1 + self.n_features_in_) * self.float_size

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
