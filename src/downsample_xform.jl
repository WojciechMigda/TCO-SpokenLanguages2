#!/usr/bin/env julia
################################################################################
#
#  Copyright (c) 2015 Wojciech Migda
#  All rights reserved
#  Distributed under the terms of the Apache 2.0 license
#
################################################################################
#
#  Filename: downsample_xform.jl
#
#  Decription:
#       Downsample MFCC frames along the time axis using fixed ratio (from ARGS)
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

using HDF5

const THIS_DIR = dirname(Base.source_path())

function downsample{_Tp<:FloatingPoint}(
    # code_native(downsample, (Int, Vector{Float64}, Vector{Float64}))
    ratio::Int,
    ivec::AbstractVector{_Tp},
    ovec::AbstractVector{_Tp})

    const STRIDE::Int = ratio
    const ILEN::Int = size(ivec, 1)

    opos::Int = 1
    ipos::Int = 1
    @inbounds @simd for ipos = 1:STRIDE:ILEN - STRIDE + 1
        ovec[opos] = 0
        @simd for sub = 0:STRIDE - 1
            ovec[opos] += ivec[ipos + sub]
        end
        ovec[opos] /= STRIDE

        opos += 1
    end
end

function transform(RATIO::Int, MFCC_H5_FNAME::ASCIIString, MFCC_H5_DSET::ASCIIString, OUT_H5_FNAME::ASCIIString, OUT_H5_DSET::ASCIIString)
    fid = h5open(MFCC_H5_FNAME, "r")
    fullX = fid[MFCC_H5_DSET]
    
    newX = Vector{Float64}[]
    
    const NCOLS = size(fullX, 2)

    res = Array(eltype(fullX), div(1001, RATIO))

    for coli = 1:NCOLS
        Xi = reshape(fullX[:, coli], (1001, 13))

        v = similar(Xi, eltype(Xi), div(size(Xi)[1], RATIO), size(Xi)[2])
        for i in 1:size(Xi)[2]
            downsample(RATIO, Xi[:, i], res)
            v[:, i] = res
        end

        push!(newX, vec(v))

        if coli % 100 == 0 println(coli) end
    end
    
    close(fid)
    
    h5write(OUT_H5_FNAME, OUT_H5_DSET, hcat(newX...))
end

function main()
    if size(ARGS)[1] != 2
        println("Specify integer downsampling RATIO and select input data (1: training, 2: test, 3: both.")
    else
        RATIO = int(ARGS[1])
        SELECT = int(ARGS[2])
        if SELECT == 1 or SELECT == 3
            OUT_H5_FNAME = "trainXscale$(RATIO).h5"
            if isfile(OUT_H5_FNAME)
                rm(OUT_H5_FNAME)
            end
            transform(RATIO, "trainXmfcc.h5", "lid/train/X/mfcc", OUT_H5_FNAME, "lid/train/X/scale$(RATIO)")
        elseif SELECT == 2 or SELECT == 3
            OUT_H5_FNAME = "testXscale$(RATIO).h5"
            if isfile(OUT_H5_FNAME)
                rm(OUT_H5_FNAME)
            end
            transform(RATIO, "testXmfcc.h5", "lid/test/X/mfcc", OUT_H5_FNAME, "lid/test/X/scale$(RATIO)")
        end
    end
end

if ~isinteractive()
    main()
end
