#!/usr/bin/env julia
################################################################################
#
#  Copyright (c) 2015 Wojciech Migda
#  All rights reserved
#  Distributed under the terms of the Apache 2.0 license
#
################################################################################
#
#  Filename: csv2h5.jl
#
#  Decription:
#       Transform CSV files into HDF5 stores
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

function main()
    const DATA_DIR = "$(THIS_DIR)/../../data"
    const CSV_DIR = DATA_DIR
    const TRAIN_CSV = "$(CSV_DIR)/trainingData.csv"
    const TEST_CSV = "$(CSV_DIR)/testingData.csv"
    
    const TRAIN = readdlm(TRAIN_CSV, ',', ASCIIString, header=true)[1]
    const TEST = readdlm(TEST_CSV, ASCIIString)

    h5write("trainXmp3.h5", "lid/train/X/mp3", TRAIN[:, 1])
    h5write("trainYmp3.h5", "lid/train/y", TRAIN[:, 2])
    h5write("testXmp3.h5", "lid/test/X/mp3", TEST[:, 1])
end

if ~isinteractive()
    main()
end
