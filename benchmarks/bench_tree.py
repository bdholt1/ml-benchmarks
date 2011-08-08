"""Tree benchmarks"""

import numpy as np
from datetime import datetime


def bench_skl(X, y, T, valid):
#
#       .. scikits.learn ..
#
    from scikits.learn import tree
    start = datetime.now()
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)
    score = np.mean(clf.predict(T) == valid)
    return score, datetime.now() - start


def bench_milk(X, y, T, valid):
#
#       .. milk ..
#
    from milk.supervised import tree
    start = datetime.now()
    learner = tree.tree_learner()
    model = learner.train(X, y)
    pred = np.sign(map(model.apply, T))
    score = np.mean(pred == valid)
    return score, datetime.now() - start


def bench_orange(X, y, T, valid):
#
#       .. Orange ..
#
    import orange
    start = datetime.now()

    # prepare data in Orange's format
    columns = []
    for i in range(0, X.shape[1]):
        columns.append("a" + str(i))
    [orange.EnumVariable(x) for x in columns]
    classValues = ['0', '1']

    domain = orange.Domain(map(orange.FloatVariable, columns),
                   orange.EnumVariable("class", values=classValues))
    y.shape = (len(y), 1) #reshape for Orange
    y[np.where(y < 0)] = 0 # change class labels to 0..K
    orng_train_data = orange.ExampleTable(domain, np.hstack((X, y)))

    valid.shape = (len(valid), 1)  #reshape for Orange
    valid[np.where(valid < 0)] = 0 # change class labels to 0..K
    orng_test_data = orange.ExampleTable(domain, np.hstack((T, valid)))

    learner = orange.C45Learner(orng_train_data)

    pred = np.empty(T.shape[0], dtype=np.int32)
    for i, e in enumerate(orng_test_data):
        pred[i] = learner(e)

    score = np.mean(pred == valid)
    return score, datetime.now() - start


if __name__ == '__main__':
    import sys
    import misc

    # don't bother me with warnings
    import warnings
    warnings.simplefilter('ignore')
    np.seterr(all='ignore')

    print __doc__ + '\n'
    if not len(sys.argv) == 2:
        print misc.USAGE % __file__
        sys.exit(-1)
    else:
        dataset = sys.argv[1]

    print 'Loading data ...'
    data = misc.load_data(dataset)

    # set sigma to something useful
    from milk.unsupervised import pdist
    sigma = np.median(pdist(data[0]))

    print 'Done, %s samples with %s features loaded into ' \
      'memory' % data[0].shape

    score, res_skl = misc.bench(bench_skl, data)
    print 'scikits.learn: mean %.2f, std %.2f' % (
        np.mean(res_skl), np.std(res_skl))
    print 'Score: %.2f\n' % score

    score, res_milk = misc.bench(bench_milk, data)
    print 'milk: mean %.2f, std %.2f' % (
        np.mean(res_milk), np.std(res_milk))
    print 'Score: %.2f\n' % score

    score, res_orange = misc.bench(bench_orange, data)
    print 'Orange: mean %.2f, std %.2f' % (
        np.mean(res_orange), np.std(res_orange))
    print 'Score: %.2f\n' % score
