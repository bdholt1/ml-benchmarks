
import numpy as np
import os

def load_data(dataset):

    f = open(os.path.dirname(__file__) + '/data/%s_train.data' % dataset)
    X = np.fromfile(f, dtype=np.float64, sep=' ')
    f.close()

    f = open(os.path.dirname(__file__) + '/data/%s_train.labels' % dataset)
    y = np.fromfile(f, dtype=np.int32, sep=' ')
    f.close()

    f = open(os.path.dirname(__file__) + '/data/%s_valid.data' % dataset)
    T = np.fromfile(f, dtype=np.float64, sep=' ')
    f.close()

    f = open(os.path.dirname(__file__) + '/data/%s_valid.labels' % dataset)
    valid = np.fromfile(f, dtype=np.float64, sep=' ')
    f.close()

    if dataset == 'madelon':
        X = X.reshape(-1, 500)
        T = T.reshape(-1, 500)
    elif dataset == 'arcene':
        X = X.reshape(-1, 10000)
        T = T.reshape(-1, 10000)

    return  X, y, T, valid


def dtime_to_seconds(dtime):
    return dtime.seconds + (dtime.microseconds * 1e-6)

def bench(func, data, n=10):
    """
    Benchmark a given function. The function is executed n times and
    its output is expected to be of type datetime.datetime.

    All values are converted to seconds and returned in an array.

    Parameters
    ----------
    func: function to benchmark

    data: tuple (X, y, T, valid) containing training (X, y) and validation (T, valid) data.

    Returns
    -------
    D : array, size=n-2
    """
    assert n > 2
    score = np.inf
    try:
        time = []
        for i in range(n):
            score, t = func(*data)
            time.append(dtime_to_seconds(t))
        # remove extremal values
        time.pop(np.argmax(time))
        time.pop(np.argmin(time))
    except Exception as detail:
        print '%s error in function %s: ' % (repr(detail), func)
        time = []
    return score, np.array(time)

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
    c = itertools.cycle('bgrcmykbgrcmyk')
    
    n_datasets = len(datasets)
    n_packages = len(packages)
    
    m, n = means.shape
    assert m == n_datasets, ValueError("means must be shape %d,%d" %
                                       (n_datasets, n_packages))
    
    m, n = stds.shape
    assert m == n_datasets, ValueError("stds must be shape %d,%d" %
                                       (n_datasets, n_packages))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ind = np.arange(n_packages)  # the x locations for the groups
    width = 0.35       # the width of the bars
    
    for i in range(n_datasets):
        rect = ax.bar(ind + i*width, means[i, :], width,
                      color=c, yerr=stds[i, :], label=datasets[i])
    
    ax.set_title('Time needed to perform train + predict (smaller is better)')
    ax.set_xticks(ind)
    ax.set_xticklabels( tuple(packages) ) 
    ax.legend()
    
    plt.show()
    plt.savefig("bench_%s.png" % (task))
    
        
def bar_plots(self):
    
    import glob
    results = glob.glob("*.pkl")
    
                    
    # for each task we will plot a graph    
    for task in self.table.keys():
        
        datasets = set()
        for result in self.table[task]:
            datasets.add(result[0])
        # convert to list
        datasets = [d for d in datasets]
        
        packages = set()
        for result in self.table[task]:
            packages.add(result[1])
        # convert to list
        packages = [p for p in packages]
    
        scores = np.zeros((len(datasets), len(packages)))
        means = np.zeros((len(datasets), len(packages)))
        stds = np.zeros((len(datasets), len(packages)))    
        for result in self.table[task]:
            d = datasets.index(result[0])
            p = packages.index(result[1])
            scores[d, p] = result[2]
            means[d, p] = result[3]
            stds[d, p] = result[4]
            
        plot_results_for_task(task, datasets, packages,
                              scores, means, stds)               
                       
def print_result(self, task, dataset, package, score, timing_results):        
    print '%s on dataset %s' %(task, dataset)
    mean = np.mean(timing_results)
    std = np.std(timing_results)
    print '%s: mean %.2f, std %.2f' % (package, mean, std)
    print 'Score: %.2f\n' % score
    
    with open('%s_%s_%s.pkl' % (task, dataset, package)) as f:
        import pickle
        pickle.dump([task, dataset, package, score, mean, std], f)


USAGE = """usage: python %s dataset

where dataset is one of {madelon, arcene}
"""