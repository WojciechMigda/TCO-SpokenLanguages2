#!/usr/bin/env julia
################################################################################
#
#  Copyright (c) 2015 Wojciech Migda
#  All rights reserved
#  Distributed under the terms of the Apache 2.0 license
#
################################################################################
#
#  Filename: jl_gen_mfcc_feat.jl
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

using ArgParse

using HDF5
@everywhere using Worker

function generate(
    MP3_DIR::String,
    MP3_H5_FNAME::String,
    MP3_H5_DSET::String,
    MFCC_H5_FNAME::String,
    MFCC_H5_DSET::String,
    NJOBS::Int,
    NMFCC::Int,
    NFILT::Int,
    ENERGY::Bool
    )
    
    const TRAIN_MP3 = MP3_DIR

    const TRAIN_X = h5read(MP3_H5_FNAME, MP3_H5_DSET)

    const TRAIN_SUBSETS = int(linspace(0, size(TRAIN_X, 1), (NJOBS + 1)))
    #const TRAIN_SUBSETS = int(linspace(0, 30, 5))
    println(TRAIN_SUBSETS)

    foo = @parallel reduce for i = 1:length(TRAIN_SUBSETS) - 1
        Worker.gen_mfcc(
            MP3_DIR,
            sub(TRAIN_X, TRAIN_SUBSETS[i] + 1:TRAIN_SUBSETS[i + 1]),
            TRAIN_SUBSETS[i] + 1,
            nfilt=NFILT,
            numcep=NMFCC,
            appendEnergy=ENERGY
            )
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

function parse_commandline(THIS_DIR::ASCIIString)
    s = ArgParseSettings()

    @add_arg_table s begin
        #"--opt1"
        #    help = "an option with an argument"
        "--njobs", "-j"
            help = "number of jobs to schedule for parallel execution"
            arg_type = Int
            default = 1
        "--nmfcc", "-m"
            help = "number of MFCCs"
            arg_type = Int
            default = 13
        "--nfilt", "-f"
            help = "number of MFCC filterbank filters"
            arg_type = Int
            default = 26
        "--energy", "-e"
            help = "replace first MFCCs with spectral energies"
            arg_type = Bool
            default = false
        "--mp3-dir"
            help = "input MP3 files folder"
            arg_type = String
            required = true
        "--h5-mp3-file"
            help = "input H5 store with MP3 filenames"
            arg_type = String
            required = true
        "--h5-mp3-dset"
            help = "input H5 dataset with MP3 filenames"
            arg_type = String
            required = true
        "--h5-mfcc-file"
            help = "output H5 store for MFCC data"
            arg_type = String
            required = true
        "--h5-mfcc-dset"
            help = "output H5 dataset for MFCC data"
            arg_type = String
            required = true

        #"arg1"
        #    help = "a positional argument"
        #    required = true
    end

    return parse_args(s)
end

function main()
    const DATA_DIR = "$(THIS_DIR)/../../data"

    parsed_args = parse_commandline(THIS_DIR)
    println("Parsed args:")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end
    
    generate(parsed_args["mp3-dir"], parsed_args["h5-mp3-file"], parsed_args["h5-mp3-dset"], parsed_args["h5-mfcc-file"], parsed_args["h5-mfcc-dset"], parsed_args["njobs"], parsed_args["nmfcc"], parsed_args["nfilt"], parsed_args["energy"])
    #generate("$(DATA_DIR)/test", "testXmp3.h5", "lid/test/X/mp3", "testXmfcc.h5", "lid/test/X/mfcc")

end

const THIS_DIR = dirname(Base.source_path())

if ~isinteractive()
    main()
end
