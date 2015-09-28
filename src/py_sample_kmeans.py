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
#      Perform per-sample KMeans clustering on MFCC data.
#      Frames with negative energies are dropped from the input dataset.
#
#      Expected input data location:
#          trainXmfcc.h5 : /lid/train/X/mfcc
#          testXmfcc.h5 : /lid/test/X/mfcc
#
#       For test data approx running time is 12 min.
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


NJOBS = 1

def transform(NCLUST, SEED, in_h5_file, in_dset, out_h5_file, out_dset):
    import h5py
    import numpy as np

    print("Running with NCLUST=", NCLUST, ", SEED=", SEED)

    fid = h5py.File(in_h5_file, 'r')
    print("H5 fid opened ", in_h5_file)

    Xdset = fid[in_dset]
    print("Xdset opened ", in_dset, Xdset.shape)

    NSAMP = Xdset.shape[0]

    from sklearn.cluster import KMeans

    outM = np.empty((NSAMP, NCLUST * 13), float)

    for isamp in range(0, NSAMP):
        X = np.reshape(Xdset[isamp, :], (13, 1001))
        X = X[:, X[0, :] >= 0.]

        if SEED == None:
            clf = KMeans(n_clusters=NCLUST, n_jobs=NJOBS)
            pass
        else:
            clf = KMeans(n_clusters=NCLUST, n_jobs=NJOBS, random_state=SEED)
            pass
        clf.fit(X.T)
        v = clf.cluster_centers_[np.lexsort((clf.cluster_centers_[:, 1], clf.cluster_centers_[:, 0]))]
        #print(v)
        outM[isamp, :] = np.ravel(v)
        #print(outM[isamp, :])

        if isamp % 10 == 0:
            print(isamp)
            pass

        pass

    fid.close()

    fid = h5py.File(out_h5_file, 'w')
    fid[out_dset] = outM
    fid.close()

    pass

def main(argv):
    if len(argv) != 3 and len(argv) != 4:
        from sys import exit
        print("Select data (1: train, 2: test, 3: both) and specify number of clusters. The last optional argument is random seed (default=None).")
        exit(1)
        pass
    else:
        OP = int(argv[1])
        NCLUST = int(argv[2])
        if len(argv) == 4:
            SEED = int(argv[3])
            pass
        else:
            SEED = None
            pass
        pass

    if OP == 1 or OP == 3:
        transform(NCLUST, SEED,
              'trainXmfcc.h5', 'lid/train/X/mfcc',
              'trainX_sample_kmeans_' + str(NCLUST) +'.h5', 'lid/train/X/sample_kmeans_' + str(NCLUST))
        pass

    if OP == 2 or OP == 3:
        transform(NCLUST, SEED,
              'testXmfcc.h5', 'lid/test/X/mfcc',
              'testX_sample_kmeans_' + str(NCLUST) +'.h5', 'lid/test/X/sample_kmeans_' + str(NCLUST))
        pass

    pass

if __name__ == "__main__":
    from sys import argv
    main(argv)
    pass
