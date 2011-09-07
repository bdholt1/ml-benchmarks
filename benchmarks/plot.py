
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
 
 
def hcat(left, right, spaces=""):
    res = []
    for l, r in zip(left,right):
        res.append(l + spaces + r)
    return res


def frame(top, bottom):
    # calculate the max length of all the strings
    max_len = len(top)
    for b in bottom:
        if len(b) > max_len:
            max_len = len(b)
    
    f = ["="*max_len]
    spaces = " "*(max_len - len(top))
    f.append(spaces + top)
    f.append("="*max_len)
    for b in bottom:
        spaces = " "*(max_len - len(b))
        f.append(spaces + b)
    f.append("="*max_len) 
    
    return f


def rst_table(task, datasets, packages, values, use_min=True):
    """Print the results in a table like this one:
    
    ============    =======     ======     ======     =======     ========    =============      ========
         Dataset     PyMVPA     Shogun        MDP     Pybrain         MLPy     scikit-learn          Milk
    ============    =======     ======     ======     =======     ========    =============      ========
         Madelon      11.52       5.63      40.48        17.5         9.47         **5.20**          5.76
         Arcene        1.30       0.39       4.87          --         1.61             0.38      **0.33**
    ============    =======     ======     ======     =======     ========    =============      ========
    """
    import math
    
    a = "Dataset"
    b = datasets
    output = frame(a, b)
    
    value_strings = []
    for v in values:
        # turn the values into strings
        vs_temp = []
        m = np.inf
        for v2 in v:
            x = float(v2)
            if not math.isnan(x) and x < m:
                m = x 
        for v2 in v:
            x = float(v2)
            if math.isnan(x) or math.isinf(x):
                vs_temp.append("--")
            elif use_min and v2 == m:
                vs_temp.append("**%.02f**" % v2)
            else:
                vs_temp.append("%.02f" % v2)
        value_strings.append(vs_temp)
    
    # transpose the value strings list of lists so that we can work on
    # columns
    values = [list(v) for v in zip(*value_strings)]
    
    for a, b in zip(packages, values):
        o = frame(a, b)
        output = hcat(output, o, " "*4)

    return output


def prepare_results(task):

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
    rst = rst_table(task, datasets, packages, means, use_min=True)
    print "Timing for ", task
    for l in rst:
        print l
    
    print
    rst = rst_table(task, datasets, packages, scores, use_min=False)
    print "Scores for ", task
    for l in rst:
        print l    
    print
    

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

    prepare_results(task)
