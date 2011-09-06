"""Logistic regression benchmarks
TODO: Shogun, anybody else ?
"""

import numpy as np
from datetime import datetime


def bench_skl(X, y, T, valid):
#
#       .. scikits.learn ..
#
    from scikits.learn import linear_model
    start = datetime.now()
    clf = linear_model.LogisticRegression()
    clf.fit(X, y)
    score = np.mean(clf.predict(T) == valid)
    return score, datetime.now() - start


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

    score, res = misc.bench(bench_skl, data)
    misc.print_result("logistic", dataset, "scikits.learn", score, res)

    misc.save_results()
