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

function main()
    const DATA_DIR = "$(THIS_DIR)/../../data"
    const CSV_DIR = DATA_DIR
    const TRAIN_CSV = "$(CSV_DIR)/trainingData.csv"
    const TRAIN_MP3 = "$(DATA_DIR)/train"

    const TRAIN = readdlm(TRAIN_CSV, ',')

    const TRAIN_SUBSETS = int(linspace(1, size(TRAIN, 1), (div(CPU_CORES, 2) + 1)))
    #const TRAIN_SUBSETS = int(linspace(1, 30, 5)) # start from 1 to skip the header row
    println(TRAIN_SUBSETS)

    foo = @parallel Worker.reduce for i = 1:length(TRAIN_SUBSETS) - 1
        #((TRAIN_SUBSETS[i] + 1, TRAIN_SUBSETS[i + 1]), Worker.gen_mfcc(sub(TRAIN, TRAIN_SUBSETS[i] + 1:TRAIN_SUBSETS[i + 1], :)))
        Worker.gen_mfcc(sub(TRAIN, TRAIN_SUBSETS[i] + 1:TRAIN_SUBSETS[i + 1], :))
    end

    println(typeof(foo))
    const NROWS = TRAIN_SUBSETS[end] - 1
    println(size(foo[][1][], 1))
    
    fid = h5open("MFCC.h5", "w")
    
    Xdset = d_create(fid, "lid/train/X/MFCC", datatype(Float64), dataspace(size(foo[][1][], 1), NROWS))

    varstring_dtype = HDF5Datatype(HDF5.h5t_copy(HDF5.H5T_C_S1))
    HDF5.h5t_set_size(varstring_dtype.id, HDF5.H5T_VARIABLE)
    #ydset = d_create(fid, "lid/train/y", varstring_dtype, dataspace((NROWS,)))
    _ydset = fill("", NROWS)

    record = 1
    for vt = 1:size(foo, 1)
        for vx = 1:size(foo[vt][1], 1)
            Xdset[:, record] = foo[vt][1][vx]
            #println(foo[vt][2][vx])
            _ydset[record] = foo[vt][2][vx]
            record += 1
        end
    end
    #ydset = _ydset

    close(Xdset)
    close(fid)

    h5write("MFCC.h5", "lid/train/y", _ydset)

end

const THIS_DIR = dirname(Base.source_path())

@windows? (
begin
    ENV["PATH"] = "$(ENV["PATH"]);$(THIS_DIR);$(THIS_DIR)/../external/bin"
    #println(ENV["PATH"])
    const LIB_MP3_DECODER = "libmp3decoder"
end
:
begin
    const LIB_MP3_DECODER = find_library(["libmp3decoder"], [THIS_DIR, "$(THIS_DIR)/../external/bin"])
end
)

if ~isinteractive()
    main()
end
