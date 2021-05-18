# logistic-PCA-tipping
This is a Python implementation of logistic PCA, as described in:
``` latex
@article{tipping1999probabilistic,
  title={Probabilistic visualisation of high-dimensional binary data},
  author={Tipping, Michael E},
  journal={Advances in neural information processing systems},
  pages={592--598},
  year={1999},
  publisher={Citeseer}
}
```
Dependencies are NumPy and Matplotlib. The implementation is in `pca.py` and a demo is in `pca_vs_logistic_pca.ipynb`. The demo scipt implements the synthetic dataset validation in Tipping's paper, additionally comparing to regular PCA.
