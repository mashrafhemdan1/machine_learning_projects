"""Mixture model based on kmeans"""
from typing import Tuple
from common import GaussianMixture
import numpy as np


def estep(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """E-step: Assigns each datapoint to the gaussian component with the
    closest mean

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples

        """
    # TODO: Add here the implementation of the E step
    n = X.shape[0]
    k = mixture.p.shape[0]
    post = np.zeros(shape=(n, k))
    for i in range(n):
        smallest_dist = float("inf")
        for j in range(k):
            dist = np.linalg.norm(mixture.mu[j, :], X[i, :])
            if dist < smallest_dist:
                smallest_dist = dist
                j_min = j
        post[i, j_min] = 1
    return post


def mstep(X: np.ndarray, post: np.ndarray) -> Tuple[GaussianMixture, float]:
    """M-step: Updates the gaussian mixture. Each cluster
    yields a component mean and variance.

    Args: X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        float: the distortion cost for the current assignment
    """
    # TODO: Add here the implementation of the M step
    mixture = GaussianMixture
    k = post.shape[1]
    n = post.shape[0]
    d = X.shape[1]
    cost = 0
    mu = np.zeros(k, d)
    var = np.zeros(k)
    p = post.sum(axis=0)
    p = p / n
    for j in range(k):
        cluster_list = X[:, j]
        cluster_list = [i for i, x in enumerate(cluster_list) if x == 1]
        cluster_sum = np.zeros((1, d))
        for i in cluster_list:
            cluster_sum += X[i, :]
        mu[j] = cluster_sum/len(cluster_list)
        cluster_sum = np.zeros((1, d))
        for i in cluster_list:
            cluster_sum += (X[i, :] - mu[j])**2
            cost += (X[i, :] - mixture.mu[j])**2
        var[j] = cluster_sum/(len(cluster_list)*d)

    return GaussianMixture(mu, var, p), float(cost)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: distortion cost of the current assignment
    """

    # TODO: Add here the implementation of the EM step of the k-means algorithm
    pre_cost = 0
    post_cost = float('inf')
    while abs(pre_cost-post_cost) > 0.0001:
        pre_cost = post_cost
        # E step
        post = estep(X, mixture)
        # M step
        mixture, post_cost = mstep(X, post)
    return mixture, post, post_cost
