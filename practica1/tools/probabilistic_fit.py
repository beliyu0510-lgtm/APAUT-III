# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 20:03:48 2026

@author: Alberto Suárez <alberto.suarez@uam.es>
"""

import numpy as np
from scipy.optimize import minimize

from typing import Callable


def maximum_likelihood_fit(
        pdf: Callable, 
        X: np.ndarray
    ) -> float:
    # [TODO]: Doctrings (including doctests) and code 
    """
    Maximum likelihood estimate for an exponencial distribution

    Args:
        pdf: Exponential pdf function
        X: Point or array of points at which to evaluate the pdf. 
           Must satisfy x >= 0.

    Returns:
        MLE of the rate parameter

    Examples:
        >>> maximum_likelihood_fit(exp_pdf, [0.0,2.0])
        np.float64(1)
    """
    X = np.asarray(X)
    if np.any(X < 0):
        raise ValueError('All the values must be superior to zero for exponencial distribution')
    
    def log_likelihood(lam):
        lam = lam[0]
        if lam <= 0:
            return np.inf
        log_like = np.sum(np.log(pdf(X,lam)))
        return -log_like
    
    res = minimize(log_likelihood,x0=[1.0])
    return res.x[0]

def maximum_posterior_fit(
        pdf: Callable,
        prior: Callable,
        X: np.ndarray
    ) -> float:
    # [TODO]: Doctrings (including doctests) and code 
    """
    Maximum posterior estimate of lambda for an exponential distribution

    Args:
        pdf: Exponential pdf function
        prior: Prior pdf function for pdf
        X: Point or array of points at which to evaluate the pdf. 
           Must satisfy x >= 0.

    Returns:
        MAP estimate of lambda

    Examples:
        >>> maximum_posterior_fit()
    """
    
    X = np.asarray(X)
    if np.any(X<0):
        raise ValueError('All the values must be superior to zero for exponencial distribution')
    def log_posterior(lam):
        lam = lam[0]
        if lam <= 0:
            return np.inf
        log_like = np.sum(np.log(pdf(X,lam)))
        log_prior = np.log(prior(lam))
        return -(log_like + log_prior)
    res = minimize(log_posterior, x0=[1.0])
    return res.x[0]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
