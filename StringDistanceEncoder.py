# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:22:38 2024

@author: cego
"""

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import TruncatedSVD
import pandas as pd

class StringDistanceEncoder(TransformerMixin, BaseEstimator):
    """ A transformer that encodes dirty categories by using n-grams and the distance
    between the strings.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to be extracted by TruncatedSVD applied to the
        n-gram matrix.
    
    metric : str, default='dice'
        The metric to use when calculating distance between the categories.
        If metric is a string, it must be one of the options allowed by scipy.spatial.distance.pdist
        for its metric parameter, or a metric listed in sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
    
    ngram_range : tuple (min_n, max_n), default=(3, 3)
        The lower and upper boundary of the range of n-values for different char
        n-grams to be extracted. All values of n such such that min_n <= n <= max_n
        will be used.
        
    lowercase : bool, default=True
        Convert all characters to lowercase before tokenizing.
        
    Attributes
    ----------
    metric : str
        Metric selected at initialization.
    
    ngram_range : tuple
        n-gram range selected at initialization.
    
    categories_ : list
        Unique categories that were saw during the `fit` method.
    
    count_vectorizer_ : class
        Respective CountVectorizer object.
        
    categories_vectorized_ : array, shape (n_samples, n_grams)
        Output of count_vectorizer_ object over categories_ list.
    
    truncated_svd : class
        Fitted TruncatedSVD object.
    """
    def __init__(self,
                 n_components=2,
                 metric="dice",
                 ngram_range=(1, 3),
                 lowercase=True):
        
        self.n_components = n_components
        self.metric = metric
        self.ngram_range= ngram_range
        self.lowercase = lowercase

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : {DataFrame}, shape (n_samples, 1)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame.")
        
        if X.shape[1] != 1:
            raise ValueError("X must have only one column.")
        
        if X.dtypes[0] not in ["object", "string"]:
            raise TypeError("Column must have object or string dtype.")
        
        X = pd.Series(X.values.squeeze()).sort_values()
        
        # we care only about the unique categories
        X_unique = np.unique(X.values)
        self.categories_ = X_unique.tolist()
        
        # fitting CountVectorizer object using ngram_range options
        self.count_vectorizer_ = CountVectorizer(ngram_range=self.ngram_range, analyzer="char",
                                                 lowercase=self.lowercase)
        self.categories_vectorized_ = self.count_vectorizer_.fit_transform(self.categories_).toarray() > 0
        
        # generating distance matrix using the selected distance metric
        dist_array = pairwise_distances(self.categories_vectorized_,
                                        metric=self.metric)
        
        # fitting and storing the TruncatedSVD object
        truncated_svd = TruncatedSVD(n_components=self.n_components,
                                     algorithm="arpack")
        truncated_svd.fit(dist_array)
        self.truncated_svd = truncated_svd
        
        return self
    

    def transform(self, X):
        """
        Parameters
        ----------
        X : {DataFrame}, shape (n_samples, 1)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_components)
            The transformed array with the distances between the categories
            dully encoded.
        """
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame.")
        
        if X.shape[1] != 1:
            raise ValueError("X must have only one column.")
        
        if X.dtypes[0] not in ["object", "string"]:
            raise TypeError("X must have object or string dtype.")
        
        check_is_fitted(self, 'truncated_svd')
        
        X = pd.Series(X.values.squeeze())
        
        # we care only about the unique categories
        X_unique = np.unique(X.values).tolist()
        
        # vectorizing X using the fitted count_vectorizer_ object
        X_unique_vectorized = self.count_vectorizer_.transform(X_unique).toarray() > 0
        
        # calculating the distances between the categories in X and the categories
        # that were already seen before
        dist_array = pairwise_distances(X_unique_vectorized,
                                        self.categories_vectorized_,
                                        metric=self.metric)
        
        # applying dimensionality reduction to the X_unique
        X_unique_transformed = self.truncated_svd.transform(dist_array)
        
        # but now we need to put X in the original shape
        X_transformed = np.zeros((len(X), X_unique_transformed.shape[-1]))
        for i, category in enumerate(X_unique):
            cond = (X == category).values
            X_transformed[cond, :] = X_unique_transformed[i, :]
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """
        Parameters
        ----------
        X : {DataFrame}, shape (n_samples, 1)
            The input samples.
        
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        
        Returns
        -------
        X_transformed : array, shape (n_samples, n_components)
            The transformed array with the distances between the categories
            dully encoded.
        """
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame.")
        
        if X.shape[1] != 1:
            raise ValueError("X must have only one column.")
        
        if X.dtypes[0] != "object":
            raise TypeError("X must have object dtype.")
        
        self.fit(X)
        return self.transform(X)