#!/usr/bin/env julia
################################################################################
#
#  Copyright (c) 2015 Wojciech Migda
#  All rights reserved
#  Distributed under the terms of the Apache 2.0 license
#
################################################################################
#
#  Filename: Worker.jl
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
#  2015-09-20   wm              Initial version
#
################################################################################

module Worker

export reduce

using MFCC

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

type Mp3Params
    rate::Uint32
    channels::Uint32
    encoding::Uint32
end

function mp3decoder!(ifname::String, osig::AbstractVector{Int16})
    # we expect 10 secs of samples, int16, stereo at 44100 Hz
    mp3params = Mp3Params(uint32(0), uint32(0), uint32(0))

    const NREAD =
        ccall((:mp3decoder, LIB_MP3_DECODER),
            Csize_t, (Ptr{Uint8}, Ptr{Void}, Csize_t, Ptr{Mp3Params}),
            ifname, osig, size(osig, 1) * sizeof(osig[1]), &mp3params)
    @assert mp3params.encoding == 0xD0 # MPG123_ENC_SIGNED_16

    return NREAD, mp3params
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

function gen_mfcc{_Tp}(MP3_DIR::ASCIIString, TRAIN_X::AbstractVector{_Tp}, start::Int)

    const DATA_DIR = "$(THIS_DIR)/../../data"
    const TRAIN_MP3 = MP3_DIR

    const NSAMP = 10 * 44100
    const NCHAN = 2
    full_sig = Array(Int16, NSAMP * NCHAN)
    sig = similar(full_sig, Float64)

    vvX = Vector{Float64}[]

    for irow = 1:size(TRAIN_X, 1)
        const NREAD, MP3PARAMS = mp3decoder!("$(TRAIN_MP3)/$(TRAIN_X[irow])", full_sig)
        @inbounds full_sig[1 + NREAD / 2:end] = 0

        const DOWN_RATIO = 4
        downsample(DOWN_RATIO, true, full_sig, sig)

        mfcc_feat = mfcc(sig, MP3PARAMS.rate / float(DOWN_RATIO))

        push!(vvX, vec(mfcc_feat))
    end

    return (vvX, start)
end

function reduce(
    lhs::(Vector{Vector{Float64}}, Int),
    rhs::(Vector{Vector{Float64}}, Int)
    )
    return [lhs, rhs]
end

function reduce(
    lhs::Vector{(Vector{Vector{Float64}}, Int)},
    rhs::(Vector{Vector{Float64}}, Int)
    )
    push!(lhs, rhs)
    return lhs
end

function reduce(
    lhs::Vector{(Vector{Vector{Float64}}, Int)},
    rhs::Vector{(Vector{Vector{Float64}}, Int)}
    )
    append!(lhs, rhs)
    return lhs
end

end # module Worker
