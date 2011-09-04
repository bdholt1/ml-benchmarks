
import numpy as np


def plot_results_for_task(task, datasets, packages, scores, means, stds):
    """Plot the results for this task, grouping by package

    task : string
        The name of the task
    datasets : list of strings, shape = [n_datasets]
        The names of the datasets
    packages : list of strings, shape = [n_packages]
    scores : array-like, shape = [n_datasets, n_packages]
        The scores of the tests
    means : array-like, shape = [n_datasets, n_packages]
        The means of the timings
    std :  array-like, shape = [n_datasets, n_packages]
        The standard deviations of the timings
    """

    import matplotlib.pyplot as plt
    import itertools
    c = itertools.cycle('bgcmykbgrcmyk')

    n_datasets = len(datasets)
    n_packages = len(packages)

    scores = np.atleast_2d(scores)
    m, n = scores.shape
    assert m == n_datasets, ValueError("scores must be shape %d,%d" %
                                       (n_datasets, n_packages))

    means = np.atleast_2d(means)
    m, n = means.shape
    assert m == n_datasets, ValueError("means must be shape %d,%d" %
                                       (n_datasets, n_packages))

    stds = np.atleast_2d(stds)
    m, n = stds.shape
    assert m == n_datasets, ValueError("stds must be shape %d,%d" %
                                       (n_datasets, n_packages))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ind = np.arange(n_packages)  # the x locations for the groups
    width = 0.35       # the width of the bars

    for i in range(n_datasets):
        rect = ax.bar(ind + i * width, means[i, :], width,
                      color=c.next(), yerr=stds[i, :],
                      ecolor='k', label=datasets[i])

    ax.set_title('Time needed to perform train + predict (smaller is better)')
    ax.set_ylabel('Seconds')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(tuple(packages))
    ax.legend()

    plt.show()
    plt.savefig("bench_%s.png" % (task))


def bar_plot(task):

    import glob
    result_files = glob.glob("%s*.results" % (task))

    datasets = []
    packages = []

    scores = []
    means = []
    stds = []

    for i, result_file in enumerate(result_files):

        with open(result_file, 'r') as f:
            import pickle
            result = pickle.load(f)

        datasets.append(result[1])

        if packages == []:
            packages = result[2]

        scores.append(result[3])
        means.append(result[4])
        stds.append(result[5])

    plot_results_for_task(task, datasets, packages,
                          scores, means, stds)

USAGE = """usage: python plot.py package

where package is one of {elasticnet, kmeans, ...}
"""

if __name__ == "__main__":
    import sys

    # don't bother me with warnings
    import warnings
    warnings.simplefilter('ignore')
    np.seterr(all='ignore')

    #print __doc__ + '\n'
    if not len(sys.argv) == 2:
        print USAGE
        sys.exit(-1)
    else:
        task = sys.argv[1]

    bar_plot(task)
