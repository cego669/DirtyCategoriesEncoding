# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:28:54 2024

@author: cego
"""

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
import pandas as pd

class StringAgglomerativeEncoder(TransformerMixin, BaseEstimator):
    """ A transformer that applies hierarchical clustering on dirty categories based on
    a given distance metric between strings.

    Parameters
    ----------
    t : scalar, default=None
        Threshold value that controls the total number of clusters. It's behavior
        varies depending on the criterion value. The default behavior corresponds
        to the maximum number of clusters.
    
    metric : str, default='dice'
        The metric to use when calculating distance between the categories.
        If metric is a string, it must be one of the options allowed by scipy.spatial.distance.pdist
        for its metric parameter, or a metric listed in sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
    
    linkage_method : str, default='average'
        A linkage method from the possible methods available on
        scipy.cluster.hierarchy.linkage object.
    
    ngram_range : tuple (min_n, max_n), default=(3, 3)
        The lower and upper boundary of the range of n-values for different char
        n-grams to be extracted. All values of n such such that min_n <= n <= max_n
        will be used.
        
    lowercase : bool, default=True
        Convert all characters to lowercase before tokenizing.
        
    criterion : str, default='maxclust'
        The criterion to use in forming clusters. Default value is the maximum
        number of clusters. Please consult possible values from
        scipy.cluster.hierarchy.fcluster object.
    
    handle_unknow : str, default='force linkage'
        Specifies the way unknown categories are handled during transform.
        - 'force linkage': force the linkage fixing the clusters and assigning
        the category to the most similar cluster using the linkage_method selected.
        - 'impute nan': ignores new categories by assigning them to nan.
        
    Attributes
    ----------
    t : scalar
        Threshold value selected at initialization.
    
    criterion : str
        Criterion selected at initialization.
    
    metric : str
        Metric selected at initialization.
    
    linkage_method : str
        Linkage method selected at initialization.
    
    ngram_range : tuple
        n-gram range selected at initialization.
    
    criterion : str
        Criterion selected at initialization.
    
    handle_unknown : bool
        The selected way at initialization by which unknown categories are handled.
    
    categories_ : list
        Unique categories that were saw during the `fit` method.
    
    count_vectorizer_ : class
        Respective CountVectorizer object.
    
    clusters_ : list
        Cluster labels that were formed during the `fit` method.
    
    Z_ : ndarray
        The hierarchical clustering encoded as a linkage matrix. It can be used
        for the dendrogram plotting.
    
    string_cluster_dict_ : dict
        Dict that maps each unique category saw during `fit` to it's corresponding
        cluster.
    """
    def __init__(self,
                 t,
                 metric="dice",
                 linkage_method="average",
                 ngram_range=(1, 3),
                 lowercase=True,
                 criterion="maxclust",
                 handle_unknown="force linkage"):
        
        linkages_supported = ["average", "complete", "single"]
        if linkage_method not in linkages_supported:
            raise ValueError(f"linkage_method must have one of the following values: \n{linkages_supported}")
        
        handle_unknown_supported = ["force linkage", "impute nan"]
        if handle_unknown not in handle_unknown_supported:
            raise ValueError(f"handle_unknown must have one of the following values: \n{handle_unknown_supported}")
        
        self.t = t
        self.criterion = criterion
        self.metric = metric
        self.linkage_method = linkage_method
        self.ngram_range= ngram_range
        self.lowercase = lowercase
        self.handle_unknown = handle_unknown

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
        
        # we care only about the unique values
        X_unique = np.unique(X.values)
        self.categories_ = X_unique.tolist()
        
        # fitting CountVectorizer object using ngram_range options
        self.count_vectorizer_ = CountVectorizer(ngram_range=self.ngram_range, analyzer="char",
                                                 lowercase=self.lowercase)
        categories_vectorized_ = self.count_vectorizer_.fit_transform(self.categories_)
        
        # generating distance matrix using the selected distance metric
        dist_array = squareform(pairwise_distances((categories_vectorized_.toarray() > 0),
                                                   metric=self.metric))
        
        # generating the linkages
        Z = linkage(dist_array,
                    method=self.linkage_method)
        self.Z_ = Z
        
        # getting respective clusters using t and criterion
        clusters = fcluster(Z,
                            t=self.t,
                            criterion="maxclust")
        self.clusters_ = list(clusters)
        
        # storing string_cluster_dict_ for predicting categories that were saw
        # during fit
        replace_dict = {X_unique[i]: clusters[i]\
                        for i in range(len(clusters))}
        self.string_cluster_dict_ = replace_dict

        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : {DataFrame}, shape (n_samples, 1)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, 1)
            The array containing the cluster labels formed during `fit`.
        """
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame.")
        
        if X.shape[1] != 1:
            raise ValueError("X must have only one column.")
        
        if X.dtypes[0] not in ["object", "string"]:
            raise TypeError("X must have object or string dtype.")
        
        check_is_fitted(self, 'string_cluster_dict_')
        
        X = pd.Series(X.values.squeeze())
        
        # we care only about the unique categories
        X_unique = np.unique(X.values).tolist()
        
        # storing unknown values, if they exist
        unknown_values = [str(value) for value in X_unique\
                          if str(value) not in self.string_cluster_dict_.keys()]
        
        # dealing appropriately with the unknown categories
        unknown_values_dict = {}
        if (len(unknown_values) != 0) and (self.handle_unknown == "force linkage"):
            
            string_cluster_df = pd.DataFrame({"string": list(self.string_cluster_dict_.keys()),
                                              "cluster": list(self.string_cluster_dict_.values())})
            
            for unknown_value in unknown_values:
                
                def get_distance(str_):
                    str1_vectorized = self.count_vectorizer_.transform([str_]).toarray() > 0
                    str2_vectorized = self.count_vectorizer_.transform([unknown_value]).toarray() > 0
                    return pairwise_distances(str1_vectorized,
                                              str2_vectorized, metric=self.metric)[0][0]
                
                string_cluster_df["dist"] = string_cluster_df["string"].apply(get_distance)
                
                if self.linkage_method == "average":
                    dist_cluster = string_cluster_df.groupby(by="cluster")["dist"].mean()
                elif self.linkage_method == "complete":
                    dist_cluster = string_cluster_df.groupby(by="cluster")["dist"].max()
                elif self.linkage_method == "single":
                    dist_cluster = string_cluster_df.groupby(by="cluster")["dist"].min()
                    
                predicted_cluster = dist_cluster.idxmin()
                
                unknown_values_dict[unknown_value] = predicted_cluster
        
        elif (len(unknown_values) != 0) and (self.handle_unknown == "impute nan"):
            
            for unknown_value in unknown_values:
                unknown_values_dict[unknown_value] = np.nan
        
        return X.replace(self.string_cluster_dict_)\
                .replace(unknown_values_dict).values.reshape(-1, 1)
    
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
        X_transformed : array, shape (n_samples, 1)
            The array containing the cluster labels formed during `fit`. Strings
            not seen during `fit` are mapped to additional clusters, as hierarchical
            clustering is not used for predicting. It is recommended to apply over
            `X_transformed` an `OneHotEncoder` object with `min_frequency` > 1.
        """
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame.")
        
        if X.shape[1] != 1:
            raise ValueError("X must have only one column.")
        
        if X.dtypes[0] != "object":
            raise TypeError("X must have object dtype.")
        
        self.fit(X)
        return self.transform(X)