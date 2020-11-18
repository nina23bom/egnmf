# coding: utf-8

# file name: egnmf.py
# Author: Takehiro Sano
# License: MIT License


import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
import pymetis
from .gnmf import GNMF, const_pNNgraph, preproc_ncw
from .metrics import calc_ac_score, calc_nmi_score


def create_hypergraph(base_clusters):
    """create the incidence matrix of base clusters' hypergraph
    
    Parameter
    ----------
    base_clusters: labels produced by base algorithms
    
    Return
    -------
    H: incidence matrix
    """
    H = None
    bc_len = base_clusters.shape[1]

    for base_cluster in base_clusters:
        bc_types = np.unique(base_cluster)
        bc_types_len = len(bc_types)
        bc2id = dict(zip(bc_types, np.arange(bc_types_len)))
        h = np.zeros((bc_len, bc_types_len))
        for i, bc_elem in enumerate(base_cluster):
            h[i, bc2id[bc_elem]] = 1.0
        if H is None:
            H = h
        else:
            H = np.hstack([H, h])

    return H


def HBGF(base_clusters, nclass=None):
    """Hybrid Bipartite Graph Formulation (HBGF) 
    
    Parameters
    ----------
    base_clusters: labels produced by base algorithms
    nclass: number of classes 
    
    Return
    -------
    labels: concensus cluster obtained from HBGF
    """
    if nclass is None:
        nclass = len(np.unique(base_clusters))

    A = create_hypergraph(base_clusters)
    rowA, colA = A.shape

    W = np.vstack([np.hstack([np.zeros((colA, colA)), A.T]), np.hstack([A, np.zeros((rowA, rowA))])])

    membership = pymetis.part_graph(nparts=nclass, adjacency=nx.Graph(W))[1]
    
    labels = membership[colA:]

    return np.array(labels)


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

        self.labels_ = HBGF(np.array(base_clusters), nclass=self.n_clusters)

        return self

    


