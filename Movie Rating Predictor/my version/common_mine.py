"""Mixture model for collaborative filtering"""
import numpy as np
from typing import NamedTuple, Tuple
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Arc


class GaussianMixture(NamedTuple):
    """Tuple holding a gaussian mixture"""
    mu: np.ndarray  # (K, d) array contains the centroids of the gaussian components
    var: np.ndarray  # (K, ) array contains the variance of each gaussian component
    p: np.ndarray  # (K, ) array contains the probability of each gaussian component

    def init(self, X: np.ndarray, K: int,
         seed: int = 0) -> np.ndarray:
        """Initializes the mixture model with random points as initial
        means and uniform assingments
        It's the first step of the kmeans

        Args:
            X: (n, d) array holding the data
            K: number of components
            seed: random seed

        Returns:
            mixture: the initialized gaussian mixture
            post: (n, K) array holding the soft counts
                for all components for all examples

        """
        # TODO: create variables
        np.random.seed(seed)
        d = X.shape[1]
        n = X.shape[0]
        max_t = np.max(X)
        min_t = np.min(X)
        self = self._replace(mu=np.random.uniform(min_t, max_t, size=(K, d)))
        self = self._replace(var=np.random.uniform(min_t, max_t, size=(K,)))
        p = np.random.uniform(min_t, max_t, size=(K,))
        self = self._replace(p=p / sum(p))
        post = np.ones((n, K))/K
        return post

    def plot(self, X: np.ndarray, post: np.ndarray,
             title: str):
        """Plots the mixture model for 2D data"""
        """This function is adapted from MITx Machine Leanring class: Movies Project"""
        _, K = post.shape

        percent = post / post.sum(axis=1).reshape(-1, 1)
        fig, ax = plt.subplots()
        ax.title.set_text(title)
        ax.set_xlim((-20, 20))
        ax.set_ylim((-20, 20))
        r = 0.25
        color = ["r", "b", "k", "y", "m", "c"]
        for i, point in enumerate(X):
            theta = 0
            for j in range(K):
                offset = percent[i, j] * 360
                arc = Arc(point,
                          r,
                          r,
                          0,
                          theta,
                          theta + offset,
                          edgecolor=color[j])
                ax.add_patch(arc)
                theta += offset
        for j in range(K):
            temp_mu = self.mu[j]
            sigma = np.sqrt(self.var[j])
            circle = Circle(temp_mu, sigma, color=color[j], fill=False)
            ax.add_patch(circle)
            legend = "mu = ({:0.2f}, {:0.2f})\n stdv = {:0.2f}".format(
                temp_mu[0], temp_mu[1], sigma)
            ax.text(temp_mu[0], temp_mu[1], legend)
        plt.axis('equal')
        plt.show()


def rmse(X, Y):
    return np.sqrt(np.mean((X-Y)**2))

def bic(X: np.ndarray, mixture: GaussianMixture,
        log_likelihood: float) -> float:
    """Computes the Bayesian Information Criterion for a
    mixture of gaussians

    Args:
        X: (n, d) array holding the data
        mixture: a mixture of spherical gaussian
        log_likelihood: the log-likelihood of the data

    Returns:
        float: the BIC for this mixture
    """
    raise NotImplementedError
