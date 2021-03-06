# EGNMF

EGNMF is a GNMF-based clustering algorithm using cluster ensembles. The algorithm stably achieves a high and robust clustering performance. 

Main Results
------------

The following results are obtained from [simulation.ipynb](https://github.com/tsano430/egnmf/blob/master/simulation.ipynb) which uses the [digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html). 

- Accuracy (AC)

<p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/60049342/107864344-a1995200-6e9e-11eb-947d-2f7a959d0dd4.png">
</p>

- Normalized Mutual Information (NMI)

<p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/60049342/107864355-c68dc500-6e9e-11eb-8892-b49c82b24b75.png">
</p>

Dependencies
------------

- numpy
- scipy
- scikit-learn
- ClusterEnsembles
- munkres
- pyitlib

References
----------

[1] https://doi.org/10.11517/pjsai.JSAI2020.0_4J2GS202

[2] D. Cai, X. He, J. Han, and T. S. Huang, Graph regularized nonnegative matrix factorization for data representation, IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 33, no. 8, pp. 1548–1560, 2010.

[3] X. Z. Fern and C. E. Brodley, Solving cluster ensemble problems by bipartite graph partitioning,  In Proceedings of the 21st International Conference on Machine Learning, p. 36, 2004.
