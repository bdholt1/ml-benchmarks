
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
             

task_string = ""
dataset_string = ""
packages = []
scores = []
means = []
stds = []
                       
def print_result(task, dataset, package, score, timing_results):        
    global task_string
    global dataset_string
    global packages
    global scores
    global means
    global stds    
    
    print '%s on dataset %s' %(task, dataset)
    mean = np.mean(timing_results)
    std = np.std(timing_results)
    print '%s: mean %.2f, std %.2f' % (package, mean, std)
    print 'Score: %.2f\n' % score

    task_string = task
    dataset_string = dataset
    packages.append(package)
    scores.append(score)
    means.append(mean)
    stds.append(std) 
    
def save_results():
    global task_string
    global dataset_string
    with open('%s_%s.results' % (task_string, dataset_string), 'w') as f:
        import pickle
        pickle.dump([task_string, dataset_string, packages,
                     scores, means, stds], f)


USAGE = """usage: python %s dataset

where dataset is one of {madelon, arcene}
"""