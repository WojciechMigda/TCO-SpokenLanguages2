#!/bin/sh

./jl_mfcc2sdc.jl -n 7 -d 1 -p 3 -k 7 \
  --nmfcc 7 \
  --h5-mfcc-file "trainX_mfcc_7.h5" \
  --h5-mfcc-dset "lid/train/X/mfcc" \
  --h5-sdc-file "trainX_sdc_7137.h5" \
  --h5-sdc-dset "lid/train/X/sdc"
