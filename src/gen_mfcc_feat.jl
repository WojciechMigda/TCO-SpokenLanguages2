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

################################################################################

"""
MFCC code ported from https://github.com/jameslyons/python_speech_features
The MIT License (MIT)
"""

function hz2mel(hz)
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * log10(1 + hz / 700.0)
end
    
function mel2hz(mel)
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10 ^ (mel / 2595.0) - 1)
end

function preemphasis(signal, coeff=0.95)
    """perform preemphasis on the input signal.
    
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """    
    return vcat(signal[1], signal[2:end] - signal[1:end - 1] * coeff)
end

function preemphasis!(signal, coeff=0.95)
    """perform preemphasis on the input signal.
    
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    """    
    for i = -length(signal):-2 signal[-i] -= signal[-i - 1] * coeff end
    """
    m = signal[1]
    @simd for i = 2:length(signal)
        n = signal[i] - m * coeff
        m = signal[i]
        signal[i] = n
    end
    """
end

function framesig(sig, frame_len, frame_step, winfunc=ones)
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.    
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    const slen = size(sig, 1)
    frame_len = int(round(frame_len))
    frame_step = int(round(frame_step))
    if slen <= frame_len
        numframes = 1
    else
        numframes = 1 + int(ceil((1.0 * slen - frame_len) / frame_step))
    end

    padlen = int((numframes - 1) * frame_step + frame_len)

    _zeros = zeros(padlen - slen)
    padsignal = vcat(sig, _zeros)

    indices =
        repeat([1:frame_len]', outer=[numframes, 1]) +
        repeat([0:frame_step:numframes * frame_step - 1]', outer=[frame_len, 1])'

    frames = padsignal[indices]
    win = repeat(winfunc(frame_len)', outer=[numframes, 1])
    return frames .* win
end

function magspec(frames, NFFT)
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT. 

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded. 
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the magnitude spectrum of the corresponding frame.
    """
    padded_frames = hcat(frames, repeat(zeros(size(frames, 1)), outer=[1, NFFT - size(frames, 2) % NFFT]))
    complex_spec = rfft(padded_frames, 2)
    #TODO magspec2 -> abs2
    return abs(complex_spec)
end

function magspec2(frames, NFFT)
    """Compute the squared magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT. 

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded. 
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the magnitude spectrum of the corresponding frame.
    """
    padded_frames = hcat(frames, repeat(zeros(size(frames, 1)), outer=[1, NFFT - size(frames, 2) % NFFT]))
    complex_spec = rfft(padded_frames, 2)
    
    return abs2(complex_spec)
end

function powspec(frames, NFFT)
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT. 

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded. 
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0 / NFFT * magspec(frames, NFFT) .^ 2
end

function powspec2(frames, NFFT)
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT. 

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded. 
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0 / NFFT * magspec2(frames, NFFT)
end

function get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=samplerate / 2)
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    @assert highfreq <= (samplerate / 2) "highfreq is greater than samplerate/2"
    
    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)

    melpoints = linspace(lowmel, highmel, nfilt + 2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = int(floor((nfft + 1) * map(mel2hz, melpoints) / samplerate)) + 1

    fbank = zeros(nfilt, div(nfft, 2) + 1)
    @simd for j in 1:nfilt
        @simd for i in int(bin[j]):int(bin[j + 1])
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        end
        @simd for i in int(bin[j + 1]):int(bin[j + 2])
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
        end
    end
    return fbank
end

function fbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
    nfilt=26, nfft=512, lowfreq=0, highfreq=samplerate / 2, preemph=0.97)
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)    
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)    
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97. 
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """          
    println("fbank enter")
    @time preemphasis!(signal, preemph)
    @time frames = framesig(signal, winlen * samplerate, winstep * samplerate)
    @time pspec = powspec2(frames, nfft)
    @time energy = sum(pspec, 2) # this stores the total energy in each frame
    @time energy[find(x -> x == 0, energy)] = eps(typeof(energy[])) # if energy is zero, we get problems with log

    @time fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
    @time feat = pspec * fb' # compute the filterbank energies
    @time feat[find(x -> x == 0, feat)] = eps(typeof(feat[])) # if feat is zero, we get problems with log

    println("fbank leave")

    return feat, energy
end

function lifter(cepstra, L=22)
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.
    
    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0
        ncoeff = size(cepstra, 2)
        lift = 1 + (L / 2) * sin(pi * [0:ncoeff - 1] / L)
        return cepstra .* lift'
    else
        # values of L <= 0, do nothing
        return cepstra
    end
end

function lifter!(cepstra, L=22)
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.
    
    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0
        ncoeff = size(cepstra, 2)
        for j = 1:size(cepstra, 2)
            lift = 1 + (L / 2) * sin(pi * (j - 1) / L)
            for i = 1:size(cepstra, 1)
                cepstra[i, j] *= lift
            end
        end
    end
end

################################################################################

function _mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
               nfilt=26, nfft=512, lowfreq=0, highfreq=samplerate/2, preemph=0.97, ceplifter=22, appendEnergy=true)
    """Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13    
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97. 
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22. 
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    println("_mfcc enter")
    @time feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph)
    @time feat = log(feat)
    @time dct!(feat, 2)
    @time feat = feat[:, 1:numcep]
    @time lifter!(feat, ceplifter)
    if appendEnergy
        @time feat[:,1] = log(energy) # replace first cepstral coefficient with log of frame energy
    end
    println("_mfcc leave")
    return feat
end
################################################################################


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
    for idx in 2:5
        println("$(idx)   $(TRAIN_MP3)/$(TRAIN[idx, 1])")
        @time const NREAD, MP3PARAMS = mp3decoder!("$(TRAIN_MP3)/$(TRAIN[idx, 1])", full_sig)
        full_sig[1 + NREAD / 2:end] = 0

        const DOWN_RATIO = 4
        @time downsample(DOWN_RATIO, true, full_sig, sig)
        #sig /= maximum(abs(sig))

        _sig = readdlm("./foo2.csv")
        @time _foo = _mfcc(_sig, MP3PARAMS.rate / float(DOWN_RATIO))
        println(_foo[5:14,:])
        """
[-36.04365338911715 -2.7816942375056187e-15 -4.1322902854678015e-15 -4.948773825620595e-15 -5.112811535936408e-15 -4.623495469039853e-15 -3.597392257957527e-15 -2.2462808275634186e-15 -8.368335071669897e-16 3.600138964809304e-16 1.120263879522089e-15 1.3087830420233828e-15 9.039024328670567e-16
 -36.04365338911715 -2.7816942375056187e-15 -4.1322902854678015e-15 -4.948773825620595e-15 -5.112811535936408e-15 -4.623495469039853e-15 -3.597392257957527e-15 -2.2462808275634186e-15 -8.368335071669897e-16 3.600138964809304e-16 1.120263879522089e-15 1.3087830420233828e-15 9.039024328670567e-16
 -36.04365338911715 -2.7816942375056187e-15 -4.1322902854678015e-15 -4.948773825620595e-15 -5.112811535936408e-15 -4.623495469039853e-15 -3.597392257957527e-15 -2.2462808275634186e-15 -8.368335071669897e-16 3.600138964809304e-16 1.120263879522089e-15 1.3087830420233828e-15 9.039024328670567e-16
 -36.04365338911715 -2.7816942375056187e-15 -4.1322902854678015e-15 -4.948773825620595e-15 -5.112811535936408e-15 -4.623495469039853e-15 -3.597392257957527e-15 -2.2462808275634186e-15 -8.368335071669897e-16 3.600138964809304e-16 1.120263879522089e-15 1.3087830420233828e-15 9.039024328670567e-16
 5.01854363329326 1.7002922966193872 -0.4609932749155671 7.115839274771968 2.0166904859617567 14.680559735072626 8.139870594749334 0.4952507178771307 -16.402947004517227 -9.591542203596502 -9.064206910294999 -1.7392328318119117 8.681381939936253
 6.253057451202818 4.689775275491571 -2.169782425332075 3.700404786770254 -7.8260704685821665 3.2138549653839843 -3.3616223215820837 14.530450864707612 -0.46284378139648136 5.055900128809202 -7.726609924090707 3.1065968846681375 -0.9296507949427201
 6.862359480564352 3.558960852829561 -2.495597592986346 6.285401892154045 -10.361812674575447 3.277860895413364 -9.539924014329376 8.91449638045395 0.31249140840785217 -6.024122538313943 -6.770259176313748 3.408454865091982 -3.8132870283127818
 6.96384936996908 2.2235512299128506 -0.6911468783552634 4.704014199582309 -13.932832529486845 -0.5416167850569376 -13.194613268804664 4.834724097968524 -2.9517881891170012 -10.809905714434256 -6.448128453863021 3.012507066061671 -6.9828107767115295
 7.027670910739485 3.331087110518866 -0.44837652002254896 3.2688040520487633 -11.245391695091751 5.4820988394593595 -10.575808876755852 -1.6028968623921778 -3.226320460472883 -4.191275390999003 -1.4888877059427994 -2.743011259008628 -4.000326512717334
 7.117967044335296 3.8751798002651183 -3.024658687780246 5.186068650426844 -11.304451805242223 -0.835453755621828 -7.084805537977053 0.9486514758429959 -6.306329734501795 -2.7710406099759615 -12.540319881346758 -10.463982367930694 -3.4940740045574143]
        """

        #@time @inbounds foo = mfcc(sig, MP3PARAMS.rate / float(DOWN_RATIO))
        @time foo = mfcc(sig, MP3PARAMS.rate / float(DOWN_RATIO))
        println(size(foo[1]))
        #println(foo[1])

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
