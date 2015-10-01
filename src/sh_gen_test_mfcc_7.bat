julia jl_gen_mfcc_feat.jl -j 6 -m 7 -f 14 -e false ^
  --mp3-dir "../../data/test" ^
  --h5-mp3-file "testXmp3.h5" ^
  --h5-mp3-dset "lid/test/X/mp3" ^
  --h5-mfcc-file "testX_mfcc_7.h5" ^
  --h5-mfcc-dset "lid/test/X/mfcc"
