import numpy as np
import kmeans_mine
import common_mine
import naive_em
import em

X = np.loadtxt('toy_data.txt');
d = X.shape[1]
# TODO: Your code here
for K in [1, 2, 3, 4]:
    min_cost = float('inf')
    mixture = common_mine.GaussianMixture(np.zeros((K, d)), np.zeros((K,)), np.zeros((K,)))
    for seed in [0, 1, 2, 3, 4]:
        post = mixture.init(X, K, seed)
        mixture, post, cost = kmeans_mine.run(X, mixture, post)
        if(cost < min_cost):
            min_cost = cost
            best_mixture = mixture
            best_post = post
    best_mixture.plot(best_post)
