#!/repo/anaconda2/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

################################################################################
#
#  Copyright (c) 2015 Wojciech Migda
#  All rights reserved
#  Distributed under the terms of the Apache 2.0 license
#
################################################################################
#
#  Filename: py_sample_kmeans.py
#
#  Decription:
#      ExtraTreesClassifier on KMeans 3-cluster
#
#  Authors:
#       Wojciech Migda
#
################################################################################
#
#  History:
#  --------
#  Date         Who  Ticket     Description
#  ----------   ---  ---------  ------------------------------------------------
#  2015-09-24   wm              Initial version
#
################################################################################

def h5read(fname, dname):
    import h5py
    import numpy as np

    fid = h5py.File(fname, 'r')
    dset = fid[dname]
    M = np.empty(dset.shape, dtype=dset.dtype)
    dset.read_direct(M)
    fid.close()
    return M

def predict(clf):
    import numpy as np

    X = h5read('testX_sample_kmeans_3.h5', 'lid/test/X/sample_kmeans_3')

    #print("Data read.")

    yprob = clf.predict_proba(X)
    mp3db = h5read('testXmp3.h5', 'lid/test/X/mp3')
    ylabels = h5read('ydict.h5', 'lid/data/y/labels')
    ylang = h5read('ydict.h5', 'lid/data/y/lang')
    ydict = {k : v for k, v in zip(ylabels, ylang)}

    top_labels = np.zeros((yprob.shape[0], 3))

    for isamp in range(0, yprob.shape[0]):
        #best = np.argmax(yprob[isamp])
        #print best
        NTOP = 3
        top_indices = np.argpartition(yprob[isamp], -NTOP)[-NTOP:]
        top_probs = yprob[isamp][top_indices]
        order = np.argsort(top_probs)
        #print(top_indices)
        #print(top_probs)
        #print(order)
        print(mp3db[isamp] + ',' + ydict[top_indices[order[2]]] + ',1')
        print(mp3db[isamp] + ',' + ydict[top_indices[order[1]]] + ',2')
        print(mp3db[isamp] + ',' + ydict[top_indices[order[0]]] + ',3')

        pass

    pass

def fit_and_predict(NEST, SEED):
    import h5py
    import numpy as np

    y = h5read('trainYlabels.h5', 'lid/train/y/labels')
    X = h5read('trainX_sample_kmeans_3.h5', 'lid/train/X/sample_kmeans_3')

    #print("Data read.")

    from sklearn.ensemble import ExtraTreesClassifier

    if SEED == None:
        clf = ExtraTreesClassifier(verbose=1, n_estimators=NEST)
        pass
    else:
        clf = ExtraTreesClassifier(verbose=1, n_estimators=NEST, random_state=SEED)
        pass

    clf.fit(X, y)
    #print("Fit done.")

    predict(clf)

    pass

def main(argv):
    if len(argv) != 2 and len(argv) != 3:
        from sys import exit
        print("Specify number of ExtraTrees estimators. The last optional argument is random seed (default=None).")
        exit(1)
        pass
    else:
        NEST = int(argv[1])
        if len(argv) == 3:
            SEED = int(argv[2])
            pass
        else:
            SEED = None
            pass

        fit_and_predict(NEST, SEED)

        pass

    pass

if __name__ == "__main__":
    from sys import argv
    main(argv)
    pass
