"""K-means clustering

Score is the inertia
"""

import numpy as np
from datetime import datetime

n_components = 9

def inertia(X, centers):
    # helper to compute inertia
    n_samples = X.shape[0]
    k = centers.shape[0]
    from scikits.learn import metrics
    distances = metrics.euclidean_distances(centers, X, None,
                                            squared=True)
    z = np.empty(n_samples, dtype=np.int)
    z.fill(-1)
    mindist = np.empty(n_samples)
    mindist.fill(np.infty)
    for q in range(k):
        dist = np.sum((X - centers[q]) ** 2, axis=1)
        z[dist < mindist] = q
        mindist = np.minimum(dist, mindist)
    inertia = mindist.sum()
    return inertia


def bench_shogun(X, y, T, valid):
#
#       .. Shogun ..
#
    from shogun.Distance import EuclidianDistance
    from shogun.Features import RealFeatures
    from shogun.Clustering import KMeans
    start = datetime.now()
    feat = RealFeatures(X.T)
    distance = EuclidianDistance(feat, feat)
    clf = KMeans(n_components, distance)
    clf.train()
    delta = datetime.now() - start
    return inertia(X, clf.get_cluster_centers().T), delta


def bench_skl(X, y, T, valid):
#
#       .. scikits.learn ..
#
    from scikits.learn import cluster as skl_cluster
    start = datetime.now()
    clf = skl_cluster.KMeans(k=n_components, n_init=1)
    clf.fit(X)
    delta = datetime.now() - start
    return inertia(X, clf.cluster_centers_), delta


def bench_pybrain(X, y, T, valid):
#
#       .. pybrain ..
#
    from pybrain.auxiliary import kmeans as pybrain_kmeans
    start = datetime.now()
    pybrain_kmeans.kmeanspp(X, n_components)
    delta = datetime.now() - start
    return np.inf, delta


def bench_mlpy(X, y, T, valid):
#
#       .. MLPy ..
#
    from mlpy import Kmeans as mlpy_Kmeans
    start = datetime.now()
    clf = mlpy_Kmeans(n_components)
    clf.compute(X)
    delta = datetime.now() - start
    return inertia(X, clf.means), delta


def bench_mdp(X, y, T, valid):
#
#       .. MDP ..
#
    from mdp.nodes import KMeansClassifier as mdp_KMeans
    start = datetime.now()
    clf = mdp_KMeans(n_components)
    clf.label(X)
    delta = datetime.now() - start
    return inertia(X, clf._centroids), delta


def bench_milk(X, y, T, valid):
#
#       .. milk ..
#
    from milk.unsupervised import kmeans as milk_kmeans
    start = datetime.now()
    _, centroids = milk_kmeans(X, n_components)
    delta = datetime.now() - start
    return inertia(X, centroids), delta


if __name__ == '__main__':
    import sys, misc

    # don't bother me with warnings
    import warnings; warnings.simplefilter('ignore')
    np.seterr(all='ignore')

    print __doc__ + '\n'
    if not len(sys.argv) == 2:
        print misc.USAGE % __file__
        sys.exit(-1)
    else:
        dataset = sys.argv[1]

    print 'Loading data ...'
    data = misc.load_data(dataset)

    print 'Done, %s samples with %s features loaded into ' \
      'memory\n' % data[0].shape

    score, res = misc.bench(bench_shogun, data)
    misc.print_result("kmeans", dataset, "Shogun", score, res)

    score, res = misc.bench(bench_mdp, data)
    misc.print_result("kmeans", dataset, "MDP", score, res)

    score, res = misc.bench(bench_skl, data)
    misc.print_result("kmeans", dataset, "scikits.learn", score, res)

    score, res = misc.bench(bench_mlpy, data)
    misc.print_result("kmeans", dataset, "MLPy", score, res)

    score, res = misc.bench(bench_pybrain, data)
    misc.print_result("kmeans", dataset, "Pybrain", score, res)

    score, res = misc.bench(bench_milk, data)
    misc.print_result("kmeans", dataset, "milk", score, res)

    misc.save_results()
