julia jl_mfcc2sdc.jl -m 6 -n 6 -d 1 -p 3 -k 6 ^
  --h5-mfcc-file "trainX_mfcc_6.h5" ^
  --h5-mfcc-dset "lid/train/X/mfcc" ^
  --h5-sdc-file "trainX_sdc_6136.h5" ^
  --h5-sdc-dset "lid/train/X/sdc"
