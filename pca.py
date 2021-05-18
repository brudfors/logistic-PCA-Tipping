import numpy as np
from numpy.linalg import (inv, solve)
from numpy import matmul as mm


def logistic_pca(X, num_components=None, num_iter=32):
    """Logistic principal component analysis (PCA).

    Parameters
    ----------
    X : (num_samples, num_dimensions) ndarray
        Data matrix.
    num_components : int, optional
        Number of PCA components.
    num_iter : int, default=32
        Number iterations for fitting model.

    Returns
    ----------
    W : (num_dimensions, num_components) ndarray
        Estimated projection matrix.
    mu : (num_components, num_samples) ndarray
        Estimated latent variables.
    b : (num_dimensions, 1) ndarray
        Estimated bias.

    Reference
    ----------
    Tipping, Michael E. "Probabilistic visualisation of high-dimensional binary data." 
    Advances in neural information processing systems (1999): 592-598.

    """
    num_samples = X.shape[0]
    num_dimensions = X.shape[1]
    num_components = _get_num_components(num_components, num_samples, num_dimensions)
    # shorthands
    N = num_samples
    D = num_dimensions
    K = num_components
    # initialise
    I = np.eye(K)
    W = np.random.randn(D, K)
    mu = np.random.randn(K, N)
    b = np.random.randn(D, 1)    
    C = np.repeat(I[:, :, np.newaxis], N, axis=2)
    xi = np.ones((N, D))  # the variational parameters
    # functions
    sig = lambda x: 1/(1 + np.exp(-x))
    lam = lambda x: (0.5 - sig(x))/(2*x)
    # fit model
    for iter in range(num_iter):
        # 1.obtain the sufficient statistics for the approximated posterior 
        # distribution of latent variables given each observation
        for n in range(N):
            # get sample
            x_n = X[n, :][:, None]
            # compute approximation
            lam_n = lam(xi[n, :])[:, None]
            # update
            C[:, :, n] = inv(I - 2*mm(W.T, lam_n*W))
            mu[:, n] = mm(C[:, :, n], mm(W.T, x_n - 0.5 + 2*lam_n*b))[:, 0]
        # 2.optimise the variational parameters in in order to make the 
        # approximation as close as possible
        for n in range(N):
            # posterior statistics
            z = mu[:, n][:, None]
            E_zz = C[:, :, n] + mm(z, z.T)
            # xi squared
            xixi = np.sum(W*mm(W, E_zz), axis=1, keepdims=True) \
                   + 2*b*mm(W, z) + b**2
            # update
            xi[n, :] = np.sqrt(np.abs(xixi[:, 0]))
        # 3.update model parameters
        E_zhzh = np.zeros((K + 1, K + 1, N))
        for n in range(N):
            z = mu[:, n][:, None]
            E_zhzh[:-1, :-1, n] = C[:, :, n] + mm(z, z.T)
            E_zhzh[:-1, -1, n] = z[:, 0]
            E_zhzh[-1, :-1, n] = z[:, 0]
            E_zhzh[-1, -1, n] = 1
        E_zh = np.append(mu, np.ones((1, N)), axis=0)
        for i in range(D):
            # compute approximation
            lam_i = lam(xi[:, i])[None][None]
            # gradient and Hessian
            H = np.sum(2*lam_i*E_zhzh, axis=2)
            g = mm(E_zh, X[:, i] - 0.5)
            # invert
            wh_i = -solve(H, g[:, None])
            wh_i = wh_i[:, 0]
            # update
            W[i, :] = wh_i[:K]
            b[i] = wh_i[K]

    return W, mu, b


def pca(X, num_components=None, zero_mean=True):
    """Principal component analysis (PCA).

    Parameters
    ----------
    X : (num_samples, num_dimensions) ndarray
        Data matrix.
    num_components : int, optional
        Number of PCA components.
    zero_mean : bool, default=True
        Zero mean data.

    Returns
    ----------
    W : (num_dimensions, num_components) ndarray
        Principal axes.        
    mu : (num_components, ) ndarray
        Principal components.    

    """
    num_samples = X.shape[0]
    num_dimensions = X.shape[1]
    num_components = _get_num_components(num_components, num_samples, num_dimensions)
    if zero_mean:        
        # zero mean
        X -= np.mean(X, axis=0)  
    # compute covariance matrix
    X = np.cov(X, rowvar=False)
    # eigen decomposition
    mu, W = np.linalg.eig(X)
    # sort descending order
    idx = np.argsort(mu)[::-1]
    W = W[:,idx]
    mu = mu[idx]
    # extract components
    mu = mu[:num_components]
    W = W[:, :num_components]

    return W, mu
    
    
def _get_num_components(num_components, num_samples, num_dimensions):
    """Get number of components (clusters).
    """
    if num_components is None:
        num_components = min(num_samples, num_dimensions)    

    return num_components    