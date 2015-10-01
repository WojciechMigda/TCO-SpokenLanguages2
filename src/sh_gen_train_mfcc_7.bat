julia jl_gen_mfcc_feat.jl -j 6 -m 7 -f 14 -e false ^
  --mp3-dir "../../data/train" ^
  --h5-mp3-file "trainXmp3.h5" ^
  --h5-mp3-dset "lid/train/X/mp3" ^
  --h5-mfcc-file "trainX_mfcc_7.h5" ^
  --h5-mfcc-dset "lid/train/X/mfcc"
