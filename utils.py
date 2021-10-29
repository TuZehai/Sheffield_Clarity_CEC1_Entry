import scipy
from scipy.signal import firwin2, stft, get_window, correlate, resample
from scipy.signal.windows import hann, hamming
from scipy.fftpack import fft, ifft
from numpy.lib.stride_tricks import as_strided
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from processor import *


def WienerFiltering(x, fs, IS=6000):
    """
    Wiener Noise Suppressor with TSNR & HRNR algorithms

    Wiener filter based on tracking a priori SNR using Decision-Directed method, proposed by Plapous et al 2006. The
    two-step noise reduction (TSNR) technique removes the annoying reverberation effect while maintaining the benefits
    of the decision-directed approach. However, classic short-time noise reduction techniques, including TSNR, introduce
    harmonic distortion in the enhanced speech. To overcome this problem, a method called harmonic regeneration noise
    reduction (HRNR)is implemented in order to refine the a priori SNR used to compute a spectral gain able to preserve
    the speech harmonics.

    Reference :   Plapous, C.; Marro, C.; Scalart, P., "Improved Signal-to-Noise Ratio Estimation for Speech Enhancement
    ", IEEE Transactions on Audio, Speech, and Language Processing, Vol. 14, Issue 6, pp. 2098 - 2108, Nov. 2006

    :param x: Noisy speech
    :param fs: Sampling rate
    :param IS: number of samples of the initial non-speech activity period or IS (i.e. Initial Silence period) at the
    beginning of the file
    :return:
    """

    l = len(x)
    s = x.copy()

    wl = int(0.025 * fs)  # window length is 25 ms
    nfft = int(2 * wl)  # fft size is twice the window length
    hanwin = hann(wl + 2)[1: -1]

    '''compute noise statistics'''
    nsum = np.zeros(nfft)
    count = 0
    for m in range(IS - wl):
        nwin = s[m: m + wl] * hanwin
        nsum = nsum + np.abs(fft(nwin, nfft)) ** 2
        count += 1
    d = nsum / count  # noise power

    '''main algorithm'''
    SP = 0.4  # Shift percentage is 60% Overlap-Add method works good with this value
    normFactor = 1 / SP
    overlap = np.fix((1 - SP) * wl)
    offset = int(wl - overlap)
    max_m = int(np.fix((l - nfft) / offset))

    zvector = np.zeros(nfft)
    oldmag = np.zeros(nfft)
    news = np.zeros(l)

    phasea = np.zeros([nfft, max_m])
    xmaga = np.zeros_like(phasea)
    tsnra = np.zeros_like(phasea)
    newmags = np.zeros_like(phasea)

    alpha = 0.99

    '''TSNR'''
    for m in range(max_m):
        begin = m * offset
        end = m * offset + wl
        speech = x[begin: end]
        winy = hanwin * speech
        ffty = fft(winy, nfft)
        phasey = np.angle(ffty)
        magy = np.abs(ffty)
        postsnr = magy ** 2 / d - 1  # calculate posteriori SNR
        postsnr = np.maximum(postsnr, 0.1)  # limitation to prevent distorsion

        # calculate a priori SNR using decision directed approach
        eta = alpha * oldmag ** 2 / d + (1 - alpha) * postsnr
        newmag = eta / (eta + 1) * magy

        # calculate TSNR
        tsnr = newmag ** 2 / d
        Gtsnr = tsnr / (tsnr + 1)
        Gtsnr = np.maximum(Gtsnr, 0.15)
        Gtsnr = gaincontrol(Gtsnr, nfft // 2)

        newmag = Gtsnr * magy
        ffty = newmag * np.exp(1j * phasey)
        oldmag = np.abs(newmag)
        news[begin: begin + nfft] = news[begin: begin + nfft] + np.real(ifft(ffty, nfft)) / normFactor
    return news


def gaincontrol(gain, constraint):
    """
    Title  : Additional Constraint on the impulse response to ensure linear convolution property

    Description :
    1- The time-duration of noisy speech frame is equal to L1 samples.

    2- This frame is then converted in the frequency domain by applying a short-time Fourier transform of size NFFT
    leading to X(wk) k=0,...,NFFT-1 when NFFT is the FFT size.

    3- The estimated noise reduction filter is G(wk) k=0,1,...,NFFT-1 leading to an equivalent impulse response
    g(n)=IFFT[G(wk)] of length L2=NFFT

    4- When applying the noise reduction filter G(wk) to the noisy speech spectrum X(wk), the multiplication
    S(wk)=G(wk)X(wk) is equivalent to a convolution in the time domain. So the time-duration of the enhanced speech s(n)
    should be equal to Ltot=L1+L2-1.

    5- If the length Ltot is greater than the time-duration of the IFFT[S(wk)] the a time-aliasing effect will appear.

    6- To overcome this phenomenon, the time-duration L2 of the equivalent impulse response g(n) should be chosen such
    that Ltot = L1 + L2 -1 <= NFFT => L2 <= NFFT+1-Ll

    here we have NFFT=2*Ll so we should have L2 <= Ll+1. I have made the following choice : the time-duration of g(n) is
    limited to L2=NFFT/2=L1

    Author : SCALART Pascal
    October  2008
    """
    meanGain = np.mean(gain ** 2)
    nfft = len(gain)
    L2 = constraint
    win = hamming(L2)

    # Frequency -> Time
    # computation of the non-constrained impulse response
    impulseR = np.real(ifft(gain))

    # application of the constraint in the time domain
    impulseR2 = np.append(np.append(impulseR[0: L2 // 2] * win[L2 // 2:], np.zeros(nfft - L2)),
                          impulseR[nfft - L2 // 2:] * win[:L2 // 2])

    # Time -> Frequency
    NewGain = np.abs(fft(impulseR2, nfft))
    meanNewGain = np.mean(NewGain ** 2)
    NewGain = NewGain * np.sqrt(meanGain / meanNewGain)

    return NewGain


def dhasp_process(wav, processor, model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        model = torch.load(model_path)
    else:
        model = torch.load(model_path, map_location='cpu')
    if processor == 'audfir':
        proc = AudiometricFIR()
    elif processor == 'convunet':
        proc = FCN()
    proc.load_state_dict(model['processor'])
    x = torch.tensor(wav, dtype=torch.float32, device=device).unsqueeze(0)
    proc_x = proc(x).cpu().detach().numpy()[0, :, :][0]
    return proc_x


def NALR(HL, nfir, fsamp):
    """
    FUnction to design an FIR NAL-R equalization filter and a flat filter having the same linear-phase time delay. The
    NAL-R filter from scipy is slightly different from matlab.
    :param HL: Hearing loss at the audiometric frequencies
    :param nfir: Order of the NAL-R EQ filter and the matching delay
    :param fsamp: Sampling rate in Hz
    :return nalr: Linear-phase filter giving the NAL-R gain function
    :return delay pure delay equal to that of the NAL-R filter
    """
    # Processing parameters
    fmax = 0.5 * fsamp

    # Audiometric frequencies
    aud = np.array([250, 500, 1000, 2000, 4000, 6000])

    # Design a flat filter having the same delay as the NAL-R filter
    delay = np.zeros(nfir + 1)
    delay[nfir // 2] = 1.0

    # Design the NAL-R filter for HI listener
    mloss = np.max(HL)
    if mloss > 0:
        # Compute the NAL-R frequency response at the audiometric frequencies
        bias = np.array([-17, -8, 1, -1, -2, -2])
        t3 = HL[1] + HL[2] + HL[3]
        if t3 <= 180:
            xave = 0.05 * t3
        else:
            xave = 9.0 + 0.116 * (t3 - 180)
        gdB = xave + 0.31 * HL + bias
        gdB = np.clip(gdB, a_min=0, a_max=None)

        # Design the linear-phase FIR filter
        fv = np.append(np.append(0, aud), fmax)  # Frequency vector for the interpolation
        cfreq = np.linspace(0, nfir, nfir + 1) / nfir  # Uniform frequency spacing from 0 to 1
        gdBv = np.append(np.append(gdB[0], gdB), gdB[-1])  # gdB vector for the interpolation
        interpf = scipy.interpolate.interp1d(fv, gdBv)
        gain = interpf(fmax * cfreq)
        glin = np.power(10, gain / 20.)
        nalr = firwin2(nfir + 1, cfreq, glin)
    else:
        nalr = delay.copy()

    return nalr, delay


def nalr_process(x, HL, nfir=140, fsamp=24000):
    nalr, _ = NALR(HL, nfir, fsamp)
    y = np.convolve(x, nalr)
    y = y[nfir // 2: -nfir // 2]
    return y
    
def nalrprocessing(x, HL, nfir=140, fsamp=24000):
    nalr, _ = NALR(HL, nfir, fsamp)
    y = np.convolve(x, nalr)
    y = y[nfir // 2: -nfir // 2]
    return y

def cal_si_snr(target, estimate):
    """
    x: target signal
    y: estimate signal
    """
    EPS = 1e-8
    pair_wise_dot = np.sum(target * estimate)
    s_target_energy = np.sum(target ** 2) + EPS
    pair_wise_proj = pair_wise_dot * target / s_target_energy
    e = estimate - pair_wise_proj
    pair_wise_si_snr = np.sum(pair_wise_proj ** 2) / (np.sum(e ** 2) + EPS)
    pair_wise_si_snr = 10 * np.log10(pair_wise_si_snr + EPS)
    return pair_wise_si_snr
    

def _check_arg(x, xname):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('%s must be one-dimensional.' % xname)
    return x


def autocorrelation(x, maxlag):
    """
    Autocorrelation with a maximum number of lags.

    `x` must be a one-dimensional numpy array.

    This computes the same result as
        numpy.correlate(x, x, mode='full')[len(x)-1:len(x)+maxlag]

    The return value has length maxlag + 1.
    """
    x = _check_arg(x, 'x')
    p = np.pad(x.conj(), maxlag, mode='constant')
    T = as_strided(p[maxlag:], shape=(maxlag+1, len(x) + maxlag),
                   strides=(-p.strides[0], p.strides[0]))
    return T.dot(p[maxlag:].conj())


def crosscorrelation(x, y, maxlag):
    """
    Cross correlation with a maximum number of lags.

    `x` and `y` must be one-dimensional numpy arrays with the same length.

    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]

    The return vaue has length 2*maxlag + 1.
    """
    x = _check_arg(x, 'x')
    y = _check_arg(y, 'y')
    py = np.pad(y.conj(), 2*maxlag, mode='constant')
    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                   strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')
    return T.dot(px)


def NALRP(HL, nfir, fsamp):
    """
    FUnction to design an FIR NAL-R equalization filter and a flat filter having the same linear-phase time delay. The
    NAL-R filter from scipy is slightly different from matlab.
    :param HL: Hearing loss at the audiometric frequencies
    :param nfir: Order of the NAL-R EQ filter and the matching delay
    :param fsamp: Sampling rate in Hz
    :return nalr: Linear-phase filter giving the NAL-R gain function
    :return delay pure delay equal to that of the NAL-R filter
    """
    # Processing parameters
    fmax = 0.5 * fsamp

    # Audiometric frequencies
    aud = np.array([250, 500, 1000, 2000, 4000, 6000])

    # Design a flat filter having the same delay as the NAL-R filter
    delay = np.zeros(nfir + 1)
    delay[nfir // 2] = 1.0

    # Design the NAL-R filter for HI listener
    mloss = np.max(HL)

    if mloss > 0:
        # Compute the NAL-R frequency response at the audiometric frequencies
        bias = np.array([-17, -8, 1, -1, -2, -2])
        t3 = HL[1] + HL[2] + HL[3]
        gdB = 0.05 * t3 + 0.31 * HL + bias

        # Correction for severe loss at 2000 Hz
        hSevere2000 = np.array([95, 100, 105, 110, 115, 120])
        gSevere = np.array([[4, 3, 0, -2, -2, -2],
                            [6, 4, 0, -3, -3, -3],
                            [8, 5, 0, -5, -5, -5],
                            [11, 7, 0, -6, -6, -6],
                            [13, 8, 0, -8, -8, -8],
                            [15, 9, 0, -9, -9, -9]]).transpose()

        if t3 > 180:
            gdB = gdB + 0.2 * (t3 - 180) / 3
        if HL[3] >= 95:
            interpf2000 = scipy.interpolate.interp1d(hSevere2000, gSevere)
            gain2000 = interpf2000(HL[3])
            gdB = gdB + gain2000

        gdB = np.clip(gdB, a_min=0, a_max=None)

        # Design the linear-phase FIR filter
        fv = np.append(np.append(0, aud), fmax)  # Frequency vector for the interpolation
        cfreq = np.linspace(0, nfir, nfir + 1) / nfir  # Uniform frequency spacing from 0 to 1
        gdBv = np.append(np.append(gdB[0], gdB), gdB[-1])  # gdB vector for the interpolation
        interpf = scipy.interpolate.interp1d(fv, gdBv)
        gain = interpf(fmax * cfreq)
        glin = np.power(10, gain / 20.)
        nalr = firwin2(nfir + 1, cfreq, glin)
    else:
        nalr = delay.copy()

    return nalr, delay


def nalrpprocessing(x, HL, nfir=140, fsamp=24000):
    nalr, _ = NALRP(HL, nfir, fsamp)
    y = np.convolve(x, nalr)
    y = y[nfir // 2: -nfir // 2]
    return y
    
    
"--------------------------------------------------------------------------------------"
"Speech enhancement performance measures https://github.com/schmiph2/pysepm"

def extract_overlapped_windows(x,nperseg,noverlap,window=None):
    # source: https://github.com/scipy/scipy/blob/v1.2.1/scipy/signal/spectral.py
    step = nperseg - noverlap
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
    strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    if window is not None:
        result = window * result
    return result


def SNRseg(clean_speech, processed_speech, fs, frameLen=0.03, overlap=0.75):
    eps=np.finfo(np.float64).eps

    winlength   = round(frameLen*fs) #window length in samples
    skiprate    = int(np.floor((1-overlap)*frameLen*fs)) #window skip in samples
    MIN_SNR     = -35 # minimum SNR in dB
    MAX_SNR     =  35 # maximum SNR in dB

    hannWin=0.5*(1-np.cos(2*np.pi*np.arange(1,winlength+1)/(winlength+1)))
    clean_speech_framed=extract_overlapped_windows(clean_speech,winlength,winlength-skiprate,hannWin)
    processed_speech_framed=extract_overlapped_windows(processed_speech,winlength,winlength-skiprate,hannWin)
    
    signal_energy = np.power(clean_speech_framed,2).sum(-1)
    noise_energy = np.power(clean_speech_framed-processed_speech_framed,2).sum(-1)
    
    segmental_snr = 10*np.log10(signal_energy/(noise_energy+eps)+eps)
    segmental_snr[segmental_snr<MIN_SNR]=MIN_SNR
    segmental_snr[segmental_snr>MAX_SNR]=MAX_SNR
    segmental_snr=segmental_snr[:-1] # remove last frame -> not valid
    return np.mean(segmental_snr)


def fwSNRseg(cleanSig, enhancedSig, fs, frameLen=0.03, overlap=0.75):
    if cleanSig.shape!=enhancedSig.shape:
        raise ValueError('The two signals do not match!')
    eps=np.finfo(np.float64).eps
    cleanSig=cleanSig.astype(np.float64)+eps
    enhancedSig=enhancedSig.astype(np.float64)+eps
    winlength   = round(frameLen*fs) #window length in samples
    skiprate    = int(np.floor((1-overlap)*frameLen*fs)) #window skip in samples
    max_freq    = fs/2 #maximum bandwidth
    num_crit    = 25# number of critical bands
    n_fft       = 2**np.ceil(np.log2(2*winlength))
    n_fftby2    = int(n_fft/2)
    gamma=0.2

    cent_freq=np.zeros((num_crit,))
    bandwidth=np.zeros((num_crit,))

    cent_freq[0]  = 50.0000;   bandwidth[0]  = 70.0000;
    cent_freq[1]  = 120.000;   bandwidth[1]  = 70.0000;
    cent_freq[2]  = 190.000;   bandwidth[2]  = 70.0000;
    cent_freq[3]  = 260.000;   bandwidth[3]  = 70.0000;
    cent_freq[4]  = 330.000;   bandwidth[4]  = 70.0000;
    cent_freq[5]  = 400.000;   bandwidth[5]  = 70.0000;
    cent_freq[6]  = 470.000;   bandwidth[6]  = 70.0000;
    cent_freq[7]  = 540.000;   bandwidth[7]  = 77.3724;
    cent_freq[8]  = 617.372;   bandwidth[8]  = 86.0056;
    cent_freq[9] =  703.378;   bandwidth[9] =  95.3398;
    cent_freq[10] = 798.717;   bandwidth[10] = 105.411;
    cent_freq[11] = 904.128;   bandwidth[11] = 116.256;
    cent_freq[12] = 1020.38;   bandwidth[12] = 127.914;
    cent_freq[13] = 1148.30;   bandwidth[13] = 140.423;
    cent_freq[14] = 1288.72;   bandwidth[14] = 153.823;
    cent_freq[15] = 1442.54;   bandwidth[15] = 168.154;
    cent_freq[16] = 1610.70;   bandwidth[16] = 183.457;
    cent_freq[17] = 1794.16;   bandwidth[17] = 199.776;
    cent_freq[18] = 1993.93;   bandwidth[18] = 217.153;
    cent_freq[19] = 2211.08;   bandwidth[19] = 235.631;
    cent_freq[20] = 2446.71;   bandwidth[20] = 255.255;
    cent_freq[21] = 2701.97;   bandwidth[21] = 276.072;
    cent_freq[22] = 2978.04;   bandwidth[22] = 298.126;
    cent_freq[23] = 3276.17;   bandwidth[23] = 321.465;
    cent_freq[24] = 3597.63;   bandwidth[24] = 346.136;


    W=np.array([0.003,0.003,0.003,0.007,0.010,0.016,0.016,0.017,0.017,0.022,0.027,0.028,0.030,0.032,0.034,0.035,0.037,0.036,0.036,0.033,0.030,0.029,0.027,0.026,
    0.026])

    bw_min=bandwidth[0]
    min_factor = np.exp (-30.0 / (2.0 * 2.303));#      % -30 dB point of filter

    all_f0=np.zeros((num_crit,))
    crit_filter=np.zeros((num_crit,int(n_fftby2)))
    j = np.arange(0,n_fftby2)


    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * (n_fftby2)
        all_f0[i] = np.floor(f0);
        bw = (bandwidth[i] / max_freq) * (n_fftby2);
        norm_factor = np.log(bw_min) - np.log(bandwidth[i]);
        crit_filter[i,:] = np.exp (-11 *(((j - np.floor(f0))/bw)**2) + norm_factor)
        crit_filter[i,:] = crit_filter[i,:]*(crit_filter[i,:] > min_factor)

    num_frames = len(cleanSig)/skiprate-(winlength/skiprate)# number of frames
    start      = 1 # starting sample
    #window     = 0.5*(1 - cos(2*pi*(1:winlength).T/(winlength+1)));


    hannWin=0.5*(1-np.cos(2*np.pi*np.arange(1,winlength+1)/(winlength+1)))
    f,t,Zxx=stft(cleanSig[0:int(num_frames)*skiprate+int(winlength-skiprate)], fs=fs, window=hannWin, nperseg=winlength, noverlap=winlength-skiprate, nfft=n_fft, detrend=False, return_onesided=True, boundary=None, padded=False)
    clean_spec=np.abs(Zxx)
    clean_spec=clean_spec[:-1,:]
    clean_spec=(clean_spec/clean_spec.sum(0))
    f,t,Zxx=stft(enhancedSig[0:int(num_frames)*skiprate+int(winlength-skiprate)], fs=fs, window=hannWin, nperseg=winlength, noverlap=winlength-skiprate, nfft=n_fft, detrend=False, return_onesided=True, boundary=None, padded=False)
    enh_spec=np.abs(Zxx)
    enh_spec=enh_spec[:-1,:]
    enh_spec=(enh_spec/enh_spec.sum(0))

    clean_energy=(crit_filter.dot(clean_spec))
    processed_energy=(crit_filter.dot(enh_spec))
    error_energy=np.power(clean_energy-processed_energy,2)
    error_energy[error_energy<eps]=eps
    W_freq=np.power(clean_energy,gamma)
    SNRlog=10*np.log10((clean_energy**2)/error_energy)
    fwSNR=np.sum(W_freq*SNRlog,0)/np.sum(W_freq,0)
    distortion=fwSNR.copy()
    distortion[distortion<-10]=-10
    distortion[distortion>35]=35

    return np.mean(distortion)
