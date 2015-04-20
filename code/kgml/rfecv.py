# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Vincent Michel <vincent.michel@inria.fr>
#          Gilles Louppe <g.louppe@gmail.com>
#          
#          Fixed errors by Oleg Razgulyaev
#
# License: BSD 3 clause

"""Recursive feature elimination for feature ranking"""
import random
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.base import MetaEstimatorMixin
from sklearn.cross_validation import check_cv
from sklearn.utils import check_arrays, safe_sqr
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.metrics.scorer import _deprecate_loss_and_score_funcs

from sklearn.externals.joblib import Parallel, delayed


class RFECVp(RFE, MetaEstimatorMixin):
    """Feature ranking with recursive feature elimination and cross-validated
       selection of the best number of features.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a `fit` method that updates a
        `coef_` attribute that holds the fitted parameters. Important features
        must correspond to high absolute values in the `coef_` array.

        For instance, this is the case for most supervised learning
        algorithms such as Support Vector Classifiers and Generalized
        Linear Models from the `svm` and `linear_model` modules.

    step : int or float, optional (default=1)
        If greater than or equal to 1, then `step` corresponds to the (integer)
        number of features to remove at each iteration.
        If within (0.0, 1.0), then `step` corresponds to the percentage
        (rounded down) of features to remove at each iteration.

    cv : int or cross-validation generator, optional (default=None)
        If int, it is the number of folds.
        If None, 3-fold cross-validation is performed by default.
        Specific cross-validation objects can also be passed, see
        `sklearn.cross_validation module` for details.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    estimator_params : dict
        Parameters for the external estimator.
        Useful for doing grid searches.

    verbose : int, default=0
        Controls verbosity of output.

    Attributes
    ----------
    `n_features_` : int
        The number of selected features with cross-validation.
    `support_` : array of shape [n_features]
        The mask of selected features.

    `ranking_` : array of shape [n_features]
        The feature ranking, such that `ranking_[i]`
        corresponds to the ranking
        position of the i-th feature.
        Selected (i.e., estimated best)
        features are assigned rank 1.

    `grid_scores_` : array of shape [n_subsets_of_features]
        The cross-validation scores such that
        `grid_scores_[i]` corresponds to
        the CV score of the i-th subset of features.

    `estimator_` : object
        The external estimator fit on the reduced dataset.

    Examples
    --------
    The following example shows how to retrieve the a-priori not known 5
    informative features in the Friedman #1 dataset.

    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.feature_selection import RFECV
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = RFECV(estimator, step=1, cv=5)
    >>> selector = selector.fit(X, y)
    >>> selector.support_ # doctest: +NORMALIZE_WHITESPACE
    array([ True,  True,  True,  True,  True,
            False, False, False, False, False], dtype=bool)
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])

    References
    ----------

    .. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
           for cancer classification using support vector machines",
           Mach. Learn., 46(1-3), 389--422, 2002.
    """
    def __init__(self, estimator, f_estimator=None, step=1, cv=None, scoring=None,
                 loss_func=None, estimator_params={}, f_estimator_params={}, verbose=0):
        self.estimator = estimator
        self.f_estimator = f_estimator
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.loss_func = loss_func
        self.estimator_params = estimator_params
        self.f_estimator_params = f_estimator_params
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the RFE model and automatically tune the number of selected
           features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the total number of features.

        y : array-like, shape = [n_samples]
            Target values (integers for classification, real numbers for
            regression).
        """ 
        if self.f_estimator == None:
            self.f_estimator = clone(self.estimator)

        X, y = check_arrays(X, y, sparse_format="csr")

        # select n_features_to_select (was always =1)
        if 0.0 < self.step < 1.0:
            step = int(self.step * X.shape[1])
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")
        n_features_to_select=1 # was: n_features_to_select=1
        
        # Initialization
        rfe = RFE(estimator=self.f_estimator, n_features_to_select=n_features_to_select,
                  step=step, estimator_params=self.f_estimator_params,
                  verbose=self.verbose - 1)

        cv = check_cv(self.cv, X, y, is_classifier(self.estimator))
        scores = np.zeros(X.shape[1])
        from collections import Counter
        rankings = Counter()
        k_max = -1
        
        # Cross-validation
        for n, (train, test) in enumerate(cv):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]

            # Compute a full ranking of the features
            ranking_ = rfe.fit(X_train, y_train).ranking_
            rankings += Counter(ranking_)
            # Score each subset of features
            for k in range(0, max(ranking_)):
                mask = np.where(ranking_ <= k + 1)[0]
                estimator = clone(self.estimator)
                estimator.fit(X_train[:, mask], y_train)

                if self.loss_func is None and self.scoring is None:
                    score = estimator.score(X_test[:, mask], y_test)
                else:
                    scorer = _deprecate_loss_and_score_funcs(
                        loss_func=self.loss_func,
                        scoring=self.scoring
                    )
                    try:
                        score = scorer(estimator, X_test[:, mask], y_test)
                    except:
                        print "**************** Except in scorer; set score=0.0 *************************"
                        score = 0.0

                if self.verbose > 0:
                    print("Finished fold with %d / %d feature ranks, score=%f"
                          % (k+1, max(ranking_), score))
                scores[k] += score
                k_max = max(k_max,k)
        n += 1
        if self.verbose > 2:
            print "rankings:",len(rankings),"k_max:",k_max
        rankings = [float(rankings[i+1]/n) for i in range(len(rankings))] 
        if self.verbose > 1:
            print "rankings:",rankings,"ranking_:",ranking_

        # Pick the best number of features on average
        scores = scores[:k_max+1]/n
        k,ibest,kbest,best_score = 0,None,None,None
        for i,s in enumerate(scores):
            k += rankings[i]
            if k >= step:
                if kbest:
                    if s > best_score:
                        ibest,kbest,best_score = i,k,s
                else:
                    ibest,kbest,best_score = i,k,s
        assert kbest != None
        if self.verbose > 0:
            print "==>Scores:",scores,"best:",best_score,"ibest:",ibest,"kbest:",kbest

        # Re-execute an elimination with best_k over the whole set
        rfe = RFE(estimator=self.f_estimator,
                  n_features_to_select=kbest,
                  step=step, estimator_params=self.f_estimator_params)

        rfe.fit(X, y)

        # Set final attributes
        self.support_ = rfe.support_
        self.n_features_ = rfe.n_features_
        self.ranking_ = rfe.ranking_
        self.estimator_ = clone(self.estimator)
        self.estimator_.set_params(**self.estimator_params)
        self.estimator_.fit(self.transform(X), y)

        self.grid_scores_ = scores
        return self

########################

from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer

from model_selection import make_grid_search

def rfe_svm(X,y,clf1="ef",clf2="ef",step=0.05,cv=None,n_estimators = 10,n_samples=0.1,verbose=0):
    """ Use RFECVp to choose features for different estimators:
        rf, ef, svm(rbf), svm(linear), lm, sgd 
    """
    if X.shape[1] < 2:
        raise ValueError("too few columns: %d"%X.shape[1])
    
    l1 = make_grid_search(clf1, X, y, n_samples=n_samples,n_estimators = n_estimators,
        kernel='rbf', verbose=verbose-1).best_estimator_

    l2 = make_grid_search(clf1, X, y, n_samples=n_samples,n_estimators = n_estimators,
        kernel='linear', verbose=verbose-1).best_estimator_
    
    if verbose>0: print "RFECV search"
    selector = RFECVp(l1,l2, step=step, cv=cv, verbose=verbose-1)
    selector = selector.fit(X, y)
    if verbose>0: print "features selected:",sum(selector.support_)
    if verbose>0: print selector.ranking_
    
    y_pred = selector.predict(X)
    if verbose>0: print "Post Score:", -mean_absolute_error(y, y_pred)
    
    return selector

#=====================

def test_vs_linear_model(N=10,M=10,informative=5):
    import random
    from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
    from sklearn.datasets.samples_generator import (make_classification,
                                                make_regression)

    from sklearn.datasets import make_friedman1
    from sklearn.datasets import make_friedman2
    from sklearn.datasets import make_friedman3
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.feature_selection.rfe import RFECV as RFECV_source
    from sklearn import grid_search, datasets
    from sklearn import svm, linear_model
    
    def run(X,y):
        alphas = [10*(0.1**i) for i in range(10)]
        lin1 = linear_model.RidgeCV(alphas=alphas)
        lin2 = RandomForestRegressor(n_estimators=10, max_depth = 12, n_jobs = -1, verbose = 0, random_state=3465343)
        lin3 = svm.SVR(kernel='linear')
        lin4 = linear_model.SGDRegressor(loss='huber', epsilon=0.1, alpha=0.0001, random_state=0)
        lin5 = svm.SVR(C=1,kernel='rbf')
        step = 0.1
        selector = rfe_svm(X,y,cv=3,step=step,verbose=2)
        est,support = selector.estimator_, selector.support_
        y_pred = est.predict(X[:,support])
        print "Test Score:", -mean_absolute_error(y, y_pred)
        """
        est = MaeRegressor(lin5,'svm')
        est2 = MaeRegressor(lin3,'svm')
        X1 = X
        for i in range(2):
            parameters = {'C':[1, 10, 100,1000,10000],'gamma':[0.1,0.01,0.001,0.0001]}        
            gs = grid_search.GridSearchCV(est, parameters, n_jobs=-1).fit(X1,y)
            print gs.best_params_
            l1 = gs.best_estimator_
            gs = grid_search.GridSearchCV(est2, parameters, n_jobs=-1).fit(X1,y)
            print gs.best_params_
            l2 = gs.best_estimator_
            selector = RFECVp(l1,l2, step=step , cv=2, verbose=0)
            selector = selector.fit(X, y)
            print "features selected:",sum(selector.support_)
            print selector.ranking_
            #selector = RFECV_source(MaeRegressor(lin), step=step, cv=3)
            #selector = selector.fit(X, y)
            #print "SOURCE features selected:",sum(selector.support_)
            y_pred = selector.predict(X)
            print "Final score:", -mean_absolute_error(y, y_pred)
            est = selector.estimator_
            X1 = X[:,selector.support_]
        """
    #for noise in [0.01,0.005]:
    for noise in [0.01]:
        print "\nLinear Problem: noise=%.2f%%\n========" % (noise*100,)
        a = np.random.sample(M)
        N2 = N*2
        X = np.reshape(np.random.sample(N2*M),(N2,M))
        y = np.dot(X,a) + np.random.sample(N2)*noise
        run(X,y)
        """

        print "\nRegression Problem: noise=%.2f%%\n========" % (noise*100,)
        X,y = make_regression(n_samples=N*2, n_features=M, n_informative=informative,
            n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, 
            noise=noise, shuffle=True, coef=False, random_state=None)
        run(X,y)
        print "\nRegression Problem, n_informative=%d effective_rank=None noise=%.2f%%\n========" % (informative,noise*100,)
        X,y = make_regression(n_samples=N*2, n_features=M, n_informative=informative,
            n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, 
            noise=noise, shuffle=True, coef=False, random_state=None)
        run(X,y)
        """
        print "\nRegression Problem, n_informative=%d effective_rank=5 noise=%.2f%%\n========" % (informative,noise*100,)
        X,y = make_regression(n_samples=N*2, n_features=M, n_informative=informative,
            n_targets=1, bias=0.0, effective_rank=5, tail_strength=0.5, 
            noise=noise, shuffle=True, coef=False, random_state=None)
        run(X,y)
        print "\nFriedman1 Problem noise=%.2f%%\n========" % (noise*100,)
        X,y = make_friedman1(n_samples=N*2, n_features=M, noise=noise, random_state=None)
        run(X,y)

def test():
    from sklearn.datasets import make_friedman1
    from sklearn.svm import SVR
    X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    estimator = SVR(kernel="linear")
    selector = RFECVp(estimator, step=1, cv=5)
    selector = selector.fit(X, y)
    print selector.support_ # doctest: +NORMALIZE_WHITESPACE
    print selector.ranking_

if __name__ == "__main__":
    random.seed(1)
    np.random.seed(0)
    import doctest
    #doctest.testmod()
    test()
    #test_vs_linear_model(N=2000,M=20,informative=3)
    #test_vs_linear_model(N=333,M=33,informative=2)
    #test_vs_linear_model(N=333,M=33,informative=17)
    #test_vs_linear_model(N=9000,M=272,informative=5)
    #test_vs_linear_model(N=3000,M=300,informative=99)
    #test_vs_linear_model(N=300,M=300,informative=99)
