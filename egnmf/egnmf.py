# coding: utf-8

# file name: egnmf.py
# Author: Takehiro Sano
# License: MIT License


import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
import ClusterEnsembles as CE
from .gnmf import GNMF, const_pNNgraph, preproc_ncw
from .metrics import calc_ac_score, calc_nmi_score


class EGNMF:
    """
    Conduct GNMF-based clustering using cluster ensembles (HBGF).
    See simulation.ipynb.
    """
    def __init__(self, n_clusters, rterm=100.0, p=5, max_iter=30, n_estimators=30, random_state=None):
        """Create a new instance"""
        if not (n_clusters > 1):
            raise(ValueError('n_components must be greater than 1.'))
        if not (rterm >= 0.0):
            raise(ValueError('rterm must be positive.'))
        if not (p > 0):
            raise(ValueError('p must be positive.'))
        if not (max_iter > 1):
            raise(ValueError('maxiter must be greater than 1.'))
        if not (n_estimators > 1):
            raise(ValueError('n_estimators must be greater than 1.'))

        self.n_clusters = n_clusters
        self.rterm = rterm
        self.p = p
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_estimators = n_estimators


    def fit(self, _X):
        """fit method"""
        base_clusters = []
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, init='random')
        W = const_pNNgraph(_X.T, self.p)
        Xncw = preproc_ncw(_X.T)

        for it in range(self.n_estimators):
            gnmf = GNMF(self.n_clusters, random_state=self.random_state+it, W=W, rterm=self.rterm, ncw=False, max_iter=self.max_iter)
            V = gnmf.fit(Xncw.T).get_coef()
            labels = kmeans.fit(V).labels_
            base_clusters.append(labels)

        self.labels_ = CE.cluster_ensembles(np.array(base_clusters), nclass=self.n_clusters, solver='hbgf')

        return self

    


