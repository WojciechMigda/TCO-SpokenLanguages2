#!/usr/bin/env julia
################################################################################
#
#  Copyright (c) 2015 Wojciech Migda
#  All rights reserved
#  Distributed under the terms of the Apache 2.0 license
#
################################################################################
#
#  Filename: jl_mfcc2sdc.jl
#
#  Decription:
#       Transform MFCCs into SDCs
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

using ArgParse
using HDF5
using DSP.TFFilter

function deltas{T<:FloatingPoint}(x::Matrix{T}, w::Int=9)
    """
    Code by David van Leeuwen (davidavdav) taken from https://github.com/davidavdav/MFCC.jl/blob/master/src/mfccs.jl
    """
    (nr, nc) = size(x)
    
    if nr == 0 || w <= 1
        return x
    end
    
    hlen = ifloor(w / 2)
    w = 2hlen + 1                 # make w odd
    win = [convert(T, hlen):-1:-hlen]
    xx = vcat(repmat(x[1, :], hlen, 1), x, repmat(x[end, :], hlen, 1)) ## take care of boundaries
    norm = 3 / (hlen * w * (hlen + 1))
    return norm * filt(TFFilter(win, [1.]), xx)[2hlen + (1:nr), :]
end

function sdc{T<:FloatingPoint}(x::Matrix{T}, n::Int=7, d::Int=1, p::Int=3, k::Int=7)
    """
    Code by David van Leeuwen (davidavdav) taken from https://github.com/davidavdav/MFCC.jl/blob/master/src/mfccs.jl
    """
    nx, nfea = size(x)
    n <= nfea || error("Not enough features for n")
    
    hnfill = (k - 1) * p / 2
    nfill1, nfill2 = ifloor(hnfill), iceil(hnfill)
    
    xx = vcat(zeros(eltype(x), nfill1, n), deltas(x[:,1:n], 2d + 1), zeros(eltype(x), nfill2, n))
    y = zeros(eltype(x), nx, n * k)
    for (dt, offset) = zip(0:p:p * k - 1, 0:n:n * k - 1)
        y[:, offset + (1:n)] = xx[dt + (1:nx), :]
    end
    
    return y
end

function transform(
    N::Int, D::Int, P::Int, K::Int,
    MFCC_H5_FNAME::ASCIIString, MFCC_H5_DSET::ASCIIString, OUT_H5_FNAME::ASCIIString, OUT_H5_DSET::ASCIIString)

    fid = h5open(MFCC_H5_FNAME, "r")
    fullX = fid[MFCC_H5_DSET]

    const NCOLS = size(fullX, 2)
    
    for coli = 1:2#NCOLS
        Xi = reshape(fullX[:, coli], (1001, 13))
        sdc_ = sdc(Xi[:, 2:13], N, D, P, K)
        println(size(Xi))
        println(size(sdc_))
        println(sdc_[1:5, :])
    end

    close(fid)

end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        #"--opt1"
        #    help = "an option with an argument"
        "--nmfcc", "-m"
            help = "number of MFCCs"
            arg_type = Int
            default = 7
        "--sdc-n", "-n"
            help = "SDC 'N' parameter"
            arg_type = Int
            default = 7
        "--sdc-d", "-d"
            help = "SDC 'D' parameter"
            arg_type = Int
            default = 1
        "--sdc-p", "-p"
            help = "SDC 'P' parameter"
            arg_type = Int
            default = 3
        "--sdc-k", "-k"
            help = "SDC 'K' parameter"
            arg_type = Int
            default = 7
        "--h5-mfcc-file"
            help = "input H5 store for MFCC data"
            arg_type = String
            required = true
        "--h5-mfcc-dset"
            help = "input H5 dataset for MFCC data"
            arg_type = String
            required = true
        "--h5-sdc-file"
            help = "output H5 store for SDC data"
            arg_type = String
            required = true
        "--h5-sdc-dset"
            help = "output H5 dataset for SDC data"
            arg_type = String
            required = true

        #"arg1"
        #    help = "a positional argument"
        #    required = true
    end

    return parse_args(s)
end

function main(args)

    parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end

    #transform(parsed_args["sdc-n"], parsed_args["sdc-d"], parsed_args["sdc-p"], parsed_args["sdc-k"],
    #    parsed_args["h5-mfcc-file"], parsed_args["h5-mfcc-dset"], parsed_args["h5-sdc-file"], parsed_args["h5-sdc-dset"])

end

if ~isinteractive()
    main(ARGS)
end
