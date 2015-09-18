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

using MFCC
using HDF5

type Mp3Params
    rate::Uint32
    channels::Uint32
    encoding::Uint32
end


function mp3decoder!(ifname::String, osig::AbstractVector{Int16})
    # we expect 10 secs of samples, int16, stereo at 44100 Hz
    mp3params = Mp3Params(uint32(0), uint32(0), uint32(0))
    const NREAD =
#        ccall((:mp3decoder, "libmp3decoder"),
        ccall((:mp3decoder, LIB_MP3_DECODER),
            Csize_t, (Ptr{Uint8}, Ptr{Void}, Csize_t, Ptr{Mp3Params}),
            ifname, osig, size(osig, 1) * sizeof(osig[1]), &mp3params)
    @assert mp3params.encoding == 0xD0 # MPG123_ENC_SIGNED_16

    return NREAD, mp3params
end

function process_range{_Tp}(inames::AbstractVector{_Tp}, istart::Int, iend::Int)
    const NELEM = iend - istart
    for idx in [istart:iend]
        println("$(idx)   $(TRAIN_MP3)/$(TRAIN[idx, 1])")
        const NSAMP = 10 * 44100
        const NCHAN = 2
        #sig = Array(Int16, NSAMP * NCHAN)
        #const NREAD, MP3PARAMS = mp3decoder!("$(TRAIN_MP3)/$(TRAIN[idx, 1])", sig)
    end
end

function downsample{_iTp, _oTp<:FloatingPoint}(
    # code_native(downsample, (Int, Bool, Vector{Int}, Vector{Float64}))
    ratio::Int,
    tomono::Bool,
    ivec::AbstractVector{_iTp},
    ovec::AbstractVector{_oTp})

    const STRIDE::Int = ratio * (tomono ? 2 : 1)
    const ILEN::Int = size(ivec, 1)

    resize!(ovec, div(ILEN, STRIDE))
    #fill!(ovec, 0)

    opos::Int = 1
    ipos::Int = 1
    @simd for ipos = 1:STRIDE:ILEN - STRIDE + 1
        @inbounds ovec[opos] = 0
        @simd for sub = 0:STRIDE - 1
            @inbounds ovec[opos] += ivec[ipos + sub]
        end
        @inbounds ovec[opos] /= STRIDE

        opos += 1
    end
end

function main()
    const DATA_DIR = "$(THIS_DIR)/../../data"
    const CSV_DIR = DATA_DIR
    const TRAIN_CSV = "$(CSV_DIR)/trainingData.csv"
    const TRAIN_MP3 = "$(DATA_DIR)/train"

    const TRAIN = readdlm(TRAIN_CSV, ',')
    const NROW = size(TRAIN, 1)

    #println(typeof(TRAIN[1, :])) # Array{Any,2}
    #println(typeof(TRAIN[:])) # Array{Any,1}
    #println(typeof(TRAIN[:, 1])) # Array{Any,1}
    #println(typeof(TRAIN[2, 1])) # SubString{ASCIIString}
    #process_range(TRAIN[:, 1], 1, 1000)
    println(TRAIN[1, 1], "|", TRAIN[1, 2])
    println(TRAIN[2, 1], "|", TRAIN[2, 2])
    println(TRAIN[NROW, 1], "|", TRAIN[NROW, 2])

    const NSAMP = 10 * 44100
    const NCHAN = 2
    full_sig = Array(Int16, NSAMP * NCHAN)
    sig = similar(full_sig, Float64)

    vvX = Vector{Float64}[]
    y = ASCIIString[]

    #for idx in 2:NROW
    for idx in 2:2
        println("$(idx)   $(TRAIN_MP3)/$(TRAIN[idx, 1])")
        @time const NREAD, MP3PARAMS = mp3decoder!("$(TRAIN_MP3)/$(TRAIN[idx, 1])", full_sig)
        full_sig[1 + NREAD / 2:end] = 0

        const DOWN_RATIO = 4
        @time downsample(DOWN_RATIO, true, full_sig, sig)
        #sig /= maximum(abs(sig))

        #@time @inbounds foo = mfcc(sig, MP3PARAMS.rate / float(DOWN_RATIO))
        @time foo = mfcc(sig, MP3PARAMS.rate / float(DOWN_RATIO), modelorder=0, lifterexp=-22, usecmp=false)
        #println(typeof(foo))
        println(size(foo[1]))
        println(foo[1])

        push!(vvX, vec(foo[1]))
        push!(y, TRAIN[idx, 2])
    end

    X = hcat(vvX...)
    #println(size(X, 1))
    #println(size(X, 2))
    #println(X[1,:])
    #println(X[:,1])
    #h5write("MFCC.h5", "lid/y", y)
    #h5write("MFCC.h5", "lid/X_MFCC", X)
end

if ~isinteractive()
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
    main()
end
