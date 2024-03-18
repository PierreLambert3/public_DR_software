#! python3
# -*-coding:Utf-8 -*

########################################################################################################
########################################################################################################

#
# %%%% !!! IMPORTANT NOTE !!! %%%%
# At the end of the fast_ms_ne.py file, a demo presents how this python code can be used. Running this file (python fast_ms_ne.py) will run the demo. Importing this module will not run the demo. The demo takes a few minutes. To be able to use the code in fast_ms_ne.py, do not forget to first compile the Cython file 'cython_implem.pyx'; check the instructions below for explanations on the required compilation steps.
# %%%% !!!                !!! %%%%

#     setup.py

# This project and the codes in this repository implement fast multi-scale neighbor embedding algorithms for nonlinear dimensionality reduction (DR).
# The fast algorithms which are implemented are described in the article "Fast Multi-scale Neighbor Embedding", from Cyril de Bodt, Dounia Mulders, Michel Verleysen and John A. Lee, currently under review in IEEE Transactions on Neural Networks and Learning Systems, in 2020.
# The implementations are provided using the python programming language, but involve some C and Cython codes for performance purposes.

# If you use the codes in this repository or the article, please cite as:
# - de Bodt, C., Mulders, D., Verleysen, M., & Lee, J. A. (2020). Fast Multi-scale Neighbor Embedding. Submitted to IEEE transactions on neural networks and learning systems. Under review.
# - BibTeX entry:
# @article{cdb2020fms,
#   title={{F}ast {M}ulti-scale {N}eighbor {E}mbedding},
#   author={de Bodt, C. and Mulders, D. and Verleysen, M. and Lee, J. A.},
#   journal={{IEEE} Trans. Neural Netw. Learn. Syst.},
#   year={2020},
#   note={Under review}
# }

# The files contained in this repository are:
# - fast_ms_ne.py: main python code to employ. At the end of the fast_ms_ne.py file, a demo presents how this python code can be used. Running this file (python fast_ms_ne.py) will run the demo. Importing this module will not run the demo. The demo takes a few minutes. To be able to use the code in fast_ms_ne.py, the Cython file 'cython_implem.pyx' must first be compiled; check the instructions below for explanations on the required compilation steps, as well as for more information on the content of the 'fast_ms_ne.py' file. The tested versions of the imported packages are also specified hereunder.
# - cython_implem.pyx: Cython implementations of the multi-scale and fast multi-scale neighbor embedding algorithms.
# - setup.py: this python file can be used to compile the Cython file 'cython_implem.pyx', which is necessary to be able to run the functions implemented in the python 'fast_ms_ne.py' file.
# - arithmetic_ansi.h, arithmetic_sse_double.h, arithmetic_sse_float.h, lbfgs.c, lbfgs.h, vptree.h: code files employed in 'cython_implem.pyx'.

# To compile the Cython file 'cython_implem.pyx' before using the python 'fast_ms_ne.py' file, perform the following steps:
# - Make sure to have Cython ('https://cython.org/', last consulted on May 30, 2020) installed on your system. For instructions, check 'https://cython.readthedocs.io/en/latest/src/quickstart/install.html' (last consulted on May 30, 2020). Note that this web link mentions that Cython requires a C compiler to be present on the system, and provides further information to get such a C compiler according to your system. Note also that Cython is available from the Anaconda Python distribution.
# - Run the command 'python setup.py build_ext --inplace' in the folder in which you downloaded the code files provided in this repository. Running this command may for instance be done from your 'Anaconda Prompt', if you are using the Anaconda Python distribution. Check 'https://cython.readthedocs.io/en/latest/src/quickstart/build.html' (last consulted on May 30, 2020) if you wish to have more information on this step.
# - You can now use the functions provided in the Python 'fast_ms_ne.py' file as you would normally do in python. You can also now run the demo of the 'fast_ms_ne.py' file, simply by running this file (python fast_ms_ne.py). The demo takes a few minutes.

# The main functions of the fast_ms_ne.py file are:
# - 'mssne': nonlinear dimensionality reduction through multi-scale SNE (Ms SNE), as presented in the reference [2] below and summarized in [1]. This function enables reducing the dimension of a data set. Given a data set with N samples, the 'mssne' function has O(N**2 log(N)) time complexity. It can hence run on databases with up to a few thousands of samples. This function is based on the Cython implementations in 'cython_implem.pyx'.
# - 'mstsne': nonlinear dimensionality reduction through multi-scale t-SNE (Ms t-SNE), as presented in the reference [6] below and summarized in [1]. This function enables reducing the dimension of a data set. Given a data set with N samples, the 'mstsne' function has O(N**2 log(N)) time complexity. It can hence run on databases with up to a few thousands of samples. This function is based on the Cython implementations in 'cython_implem.pyx'.
# - 'fmssne': nonlinear dimensionality reduction through fast multi-scale SNE (FMs SNE), as presented in the reference [1] below. This function enables reducing the dimension of a data set. Given a data set with N samples, the 'fmssne' function has O(N (log(N))**2) time complexity. It can hence run on very large-scale databases. This function is based on the Cython implementations in 'cython_implem.pyx'.
# - 'fmstsne': nonlinear dimensionality reduction through fast multi-scale t-SNE (FMs t-SNE), as presented in the reference [1] below. This function enables reducing the dimension of a data set. Given a data set with N samples, the 'fmstsne' function has O(N (log(N))**2) time complexity. It can hence run on very large-scale databases. This function is based on the Cython implementations in 'cython_implem.pyx'.
# - 'eval_dr_quality': unsupervised evaluation of the quality of a low-dimensional embedding, as introduced in [3, 4] and employed and summarized in [1, 2, 5]. This function enables computing DR quality assessment criteria measuring the neighborhood preservation from the high-dimensional space to the low-dimensional one. The documentation of the function explains the meaning of the criteria and how to interpret them. Given a data set with N samples, the 'eval_dr_quality' function has O(N**2 log(N)) time complexity. It can hence run using databases with up to a few thousands of samples. This function is not based on the Cython implementations in 'cython_implem.pyx'.
# - 'red_rnx_auc': this function is similar to the 'eval_dr_quality' function, but given a data set with N samples, the 'red_rnx_auc' function has O(N*Kup*log(N)) time complexity, where Kup is the maximum neighborhood size accounted when computing the quality criteria. This function can hence run using much larger databases than 'eval_dr_quality', provided that Kup is small compared to N. This function is based on the Cython implementations in 'cython_implem.pyx'.
# - 'viz_2d_emb' and 'viz_qa': visualization of a 2-D embedding and of the quality criteria. These functions respectively enable to:
# ---> 'viz_2d_emb': plot a 2-D embedding. This function is not based on the Cython implementations in 'cython_implem.pyx'.
# ---> 'viz_qa': depict the quality criteria computed by 'eval_dr_quality' and 'red_rnx_auc'. This function is not based on the Cython implementations in 'cython_implem.pyx'.
# The documentations of the functions describe their parameters. The demo shows how they can be used.

# Notations:
# - DR: dimensionality reduction.
# - HD: high-dimensional.
# - LD: low-dimensional.
# - HDS: HD space.
# - LDS: LD space.
# - SNE: stochastic neighbor embedding.
# - t-SNE: t-distributed SNE.
# - Ms SNE: multi-scale SNE.
# - Ms t-SNE: multi-scale t-SNE.
# - BH t-SNE: Barnes-Hut t-SNE.

# References:
# [1] de Bodt, C., Mulders, D., Verleysen, M., & Lee, J. A. (2020). Fast Multi-scale Neighbor Embedding. Submitted to IEEE transactions on neural networks and learning systems. Under review.
# [2] Lee, J. A., Peluffo-Ordóñez, D. H., & Verleysen, M. (2015). Multi-scale similarities in stochastic neighbour embedding: Reducing dimensionality while preserving both local and global structure. Neurocomputing, 169, 246-261.
# [3] Lee, J. A., & Verleysen, M. (2009). Quality assessment of dimensionality reduction: Rank-based criteria. Neurocomputing, 72(7-9), 1431-1443.
# [4] Lee, J. A., & Verleysen, M. (2010). Scale-independent quality criteria for dimensionality reduction. Pattern Recognition Letters, 31(14), 2248-2257.
# [5] Lee, J. A., Renard, E., Bernard, G., Dupont, P., & Verleysen, M. (2013). Type 1 and 2 mixtures of Kullback–Leibler divergences as cost functions in dimensionality reduction based on similarity preservation. Neurocomputing, 112, 92-108.
# [6] de Bodt, C., Mulders, D., Verleysen, M., & Lee, J. A. (2018). Perplexity-free t-SNE and twice Student tt-SNE. In ESANN (pp. 123-128).
# [7] van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine Learning Research, 9(Nov), 2579-2605.
# [8] van der Maaten, L. (2014). Accelerating t-SNE using tree-based algorithms. Journal of Machine Learning Research, 15(1), 3221-3245.

# author: Cyril de Bodt (ICTEAM - UCLouvain)
# @email: cyril __dot__ debodt __at__ uclouvain.be
# Last modification date: May 30th, 2020
# Copyright (c) 2020 Université catholique de Louvain (UCLouvain), ICTEAM. All rights reserved.

# The codes in this repository were tested with Python 3.6.5 (Anaconda distribution, Continuum Analytics, Inc.). They use the following modules:
# - numpy: version 1.14.2 tested
# - numba: version 0.37.0 tested
# - scipy: version 1.0.1 tested
# - matplotlib: version 2.2.2 tested
# - scikit-learn: version 0.19.1 tested
# - Cython: version 0.28.1 tested

# You can use, modify and redistribute this software freely, but not for commercial purposes.
# The use of this software is at your own risk; the authors are not responsible for any damage as a result from errors in the software.

########################################################################################################
########################################################################################################

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [Extension(
                "cython_implem",
                sources=["cython_implem.pyx", "lbfgs.c"],
                annotate=False,
                extra_compile_args=["-fopenmp"],
                extra_link_args=["-fopenmp"]
            )]


setup(
    name = "cython_implem",
    ext_modules = cythonize(extensions),
    include_dirs=[np.get_include(), '.']
)
