#!/usr/bin/env julia
################################################################################
#
#  Copyright (c) 2015 Wojciech Migda
#  All rights reserved
#  Distributed under the terms of the Apache 2.0 license
#
################################################################################
#
#  Filename: gen_mfcc_feat.jl
#
#  Decription:
#       Generate MFCC feature vectors
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
#  2015-09-16   wm              Initial version
#
################################################################################

addprocs(div(CPU_CORES, 2))

using HDF5
@everywhere using Worker

function generate(
    MP3_DIR::ASCIIString,
    MP3_H5_FNAME::ASCIIString,
    MP3_H5_DSET::ASCIIString,
    MFCC_H5_FNAME::ASCIIString,
    MFCC_H5_DSET::ASCIIString)
    
    const TRAIN_MP3 = MP3_DIR

    const TRAIN_X = h5read(MP3_H5_FNAME, MP3_H5_DSET)

    const TRAIN_SUBSETS = int(linspace(0, size(TRAIN_X, 1), (div(CPU_CORES, 2) + 1)))
    #const TRAIN_SUBSETS = int(linspace(0, 30, 5))
    println(TRAIN_SUBSETS)

    foo = @parallel reduce for i = 1:length(TRAIN_SUBSETS) - 1
        Worker.gen_mfcc(
            MP3_DIR,
            sub(TRAIN_X, TRAIN_SUBSETS[i] + 1:TRAIN_SUBSETS[i + 1]),
            TRAIN_SUBSETS[i] + 1)
    end

    gc()
    
    println(typeof(foo))
    const NROWS = TRAIN_SUBSETS[end]
    println(size(foo[][1][], 1))
    
    fid = h5open(MFCC_H5_FNAME, "w")
    
    Xdset = d_create(fid, MFCC_H5_DSET, datatype(Float64), dataspace(size(foo[][1][], 1), NROWS))

    for vt = 1:size(foo, 1)
        pos = foo[vt][2]
        for vx = 1:size(foo[vt][1], 1)
            Xdset[:, pos] = foo[vt][1][vx]
            pos += 1
        end
        
        gc()
    end

    gc()
    
    close(Xdset)
    close(fid)
end

function main()
    const DATA_DIR = "$(THIS_DIR)/../../data"
    
    generate("$(DATA_DIR)/train", "trainXmp3.h5", "lid/train/X/mp3", "trainXmfcc.h5", "lid/train/X/mfcc")
    generate("$(DATA_DIR)/test", "testXmp3.h5", "lid/test/X/mp3", "testXmfcc.h5", "lid/test/X/mfcc")

end

const THIS_DIR = dirname(Base.source_path())

if ~isinteractive()
    main()
end
