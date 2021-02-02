# EGNMF

EGNMF is a GNMF-based clustering algorithm using cluster ensembles. The algorithm stably achieves a high and robust clustering performance. The usage of the algorithm is shown in [simulation.ipynb][simulation]. 

Dependencies
------------

- numpy
- scipy
- scikit-learn
- networkx 
- ClusterEnsembles
- munkres
- pyitlib

References
----------

[1] https://doi.org/10.11517/pjsai.JSAI2020.0_4J2GS202

[2] D. Cai, X. He, J. Han, and T. S. Huang, Graph regularized nonnegative matrix factorization for data representation, IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 33, no. 8, pp. 1548–1560, 2010.

[3] X. Z. Fern and C. E. Brodley, Solving cluster ensemble problems by bipartite graph partitioning,  In Proceedings of the 21st International Conference on Machine Learning, p. 36, 2004.

[simulation]: https://github.com/tsano430/egnmf/blob/master/simulation.ipynb
