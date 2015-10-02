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
#      KNN classifier with StandardScaler and PCA on KMeans 3-cluster
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
#  2015-10-01   wm              Initial version
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

def predict(clf, clf_et, X, X_et):
    import numpy as np

    #print("Data read.")

    yprob = clf.predict_proba(X)

    yprob += 5. * clf_et.predict_proba(X_et)

    mp3db = h5read('testXmp3.h5', 'lid/test/X/mp3')
    ylabels = h5read('ydict.h5', 'lid/data/y/labels')
    ylang = h5read('ydict.h5', 'lid/data/y/lang')
    ydict = {k : v for k, v in zip(ylabels, ylang)}

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

def fit_and_predict(NNGB, SEED):
    import h5py
    import numpy as np

    NEST = 100

    y = h5read('trainYlabels.h5', 'lid/train/y/labels')
    X = h5read('trainX_sample_kmeans_3_raw.h5', 'lid/train/X/sample_kmeans_3')

    #print("Data read.")

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import ExtraTreesClassifier

    clf = KNeighborsClassifier(n_neighbors=NNGB)
    if SEED == None:
        clf_et = ExtraTreesClassifier(verbose=1, n_estimators=NEST)
        pass
    else:
        clf_et = ExtraTreesClassifier(verbose=1, n_estimators=NEST, random_state=SEED)
        pass

    clf_et.fit(X, y)
    print("Fit 1 done.")

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    ss_clf = StandardScaler()
    X = ss_clf.fit_transform(X)
    pca_clf = PCA(whiten=True)
    X = pca_clf.fit_transform(X)

    clf.fit(X, y)
    print("Fit 2 done.")

    Xt = h5read('testX_sample_kmeans_3_raw.h5', 'lid/test/X/sample_kmeans_3')
    Xt_et = Xt
    Xt = ss_clf.transform(Xt)
    Xt = pca_clf.transform(Xt)

    predict(clf, clf_et, Xt, Xt_et)

    pass

def main(argv):
    if len(argv) != 2 and len(argv) != 3:
        from sys import exit
        print("Specify number of KNN neighbours. The last optional argument is random seed (default=None).")
        exit(1)
        pass
    else:
        NCOMP = int(argv[1])
        if len(argv) == 3:
            SEED = int(argv[2])
            pass
        else:
            SEED = None
            pass

        fit_and_predict(NCOMP, SEED)

        pass

    pass

if __name__ == "__main__":
    from sys import argv
    main(argv)
    pass
