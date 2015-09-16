#!/usr/bin/julia
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

sig = Array(Int16, 18000)
println(size(sig, 1))

type Mp3Params
  rate::Uint32
  channels::Uint32
  encoding::Uint32
end

mp3Params = Mp3Params(uint32(0), uint32(0), uint32(0))

res = ccall((:mp3decoder, "./libmp3decoder"), Csize_t, (Ptr{Int8}, Ptr{Uint8}, Csize_t, Ptr{Mp3Params}), "../../data/0.mp3", sig, size(sig, 1) * 2, &mp3Params)


println(res)
println("rate: ", mp3Params.rate, ", channels: ", mp3Params.channels, ", encoding: ", mp3Params.encoding)
#println(sig)

#train = readdlm("D:/Workspace/TCO-SpokenLanguages2/data/trainingData.csv", ',')
train = readdlm("/repo/topcoder/SpokenLanguages2/data/trainingData.csv", ',')
NROW = size(train, 1)

println(train[1, 1], "|", train[1, 2])
println(train[2, 1], "|", train[2, 2])
println(train[NROW, 1], "|", train[NROW, 2])

for idx in [2:NROW]
    #println(x)
end
