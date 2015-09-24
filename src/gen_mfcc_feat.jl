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

addprocs(div(CPU_CORES, 1))

using HDF5
@everywhere using Worker

function main()
    const DATA_DIR = "$(THIS_DIR)/../../data"
    const TRAIN_MP3 = "$(DATA_DIR)/train"

    const TRAIN_X = h5read("trainXmp3.h5", "lid/train/X/mp3")

    #const TRAIN_SUBSETS = int(linspace(0, size(TRAIN, 1), (div(CPU_CORES, 2) + 1)))
    const TRAIN_SUBSETS = int(linspace(0, 30, 5))
    println(TRAIN_SUBSETS)

    foo = @parallel reduce for i = 1:length(TRAIN_SUBSETS) - 1
        Worker.gen_mfcc(
            sub(TRAIN_X, TRAIN_SUBSETS[i] + 1:TRAIN_SUBSETS[i + 1]),
            TRAIN_SUBSETS[i] + 1)
    end

    println(typeof(foo))
    const NROWS = TRAIN_SUBSETS[end]
    println(size(foo[][1][], 1))
    
    fid = h5open("trainXmfcc.h5", "w")
    
    Xdset = d_create(fid, "lid/train/X/mfcc", datatype(Float64), dataspace(size(foo[][1][], 1), NROWS))

    for vt = 1:size(foo, 1)
        pos = foo[vt][2]
        for vx = 1:size(foo[vt][1], 1)
            Xdset[:, pos] = foo[vt][1][vx]
            pos += 1
        end
    end

    close(Xdset)
    close(fid)
end

const THIS_DIR = dirname(Base.source_path())

if ~isinteractive()
    main()
end
