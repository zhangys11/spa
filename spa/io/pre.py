#### Pre-processing ####
from scipy import signal, sparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import shift
from statsmodels.tsa.stattools import ccovf

############# feature filter by threshold ##############
def x_thresholding(X, threshold = 10):
    '''
    All features below the threshold will be removed.
    Typically used in TOF-MS data.
    Signals weaker than 10 is mostly considered as background noise.

    Note
    ----
    This function can only be used in the original unscaled data. 
    '''
    NX = X.copy()
    NX[NX <= threshold] = 0
    return NX

############# row-wise normalization / scaling ##############

def x_normalize(X, flavor = 'highest_peak', target_max = 1000):
    '''
    Normalize each data / row by its highest peak or row vector.
    Typically used in TOF-MS data. 
    
    Parameters
    ----------
    flavor : 'highest_peak' or 'row_vector'
    target_max : the maximum value after normalization. Default is 1000.
    '''
    NX = []
    for x in X:
        if flavor == 'row_vector' or flavor == 'rowvec':
            x_max = np.linalg.norm(x)
        else:
            x_max = np.abs(x).max()
        nx = x / x_max * target_max
        NX.append(nx)
    return np.array(NX)

################### binning / kernel / sliding window ###################

def x_binning(X, X_names = None, target_dim = 0.1, flavor = 'max', display = False):
    '''
    For very high-dimensional data, such as TOF-MS, we split the features into several equal segments and do integral or max for each segment.
    We can use this function as a down-sampling / DR algorithm.

    Parameters
    ----------
    target_dim : float, range (0,1]. target_dim = 1 will return the original X.
    flavor : how do we condense each segment
        'max' - take the maximum. 
        'sum' / 'rect' / 'mean' - take the sum / integral / mean.
        'tri' - triangle integral, use a mask e.g., [1,2,3,4,3,2,1]
    display : whether show the 1st sample after processing

    Return
    ------
    NX : the new dataset matrix. size is (m, target_dim)
    NX_names : the corresponding X headers.

    '''
    
    n = X.shape[1]
    if target_dim >= 1 or target_dim <= 0:
        target_dim = n
        return X, X_names
    
    target_dim = int(target_dim * n)
    
    NX = []
    NX_names = []
    for idx, a in enumerate( np.array_split(X, target_dim, axis = 1) ):
        if X_names is not None and len(X_names) == n:
            NX_names.append( X_names[ int (a.shape[1] * (idx + 0.5)) ] ) # take the middle xlabel
        
        if flavor == 'max':
            NCol = a.max(axis = 1)
        elif flavor == 'tri':
            mask = list( range(1, int (a.shape[1] / 2) + 1) ) + list( range(a.shape[1] - int (a.shape[1] / 2) + 1, 1, -1 ) )
            NCol = a @ (np.array(mask).reshape(-1))
        else: # sum / rect /mean
            NCol = a.sum(axis = 1)
        NX.append( NCol )
    
    NX = np.array(NX).T

    if display:
        plt.plot(NX_names, NX[0])
        plt.title('New Sample #1')
        plt.yticks([]) # we only care about y's relative values.
        plt.show()

    return NX, NX_names if len(NX_names) > 0 else None

################### digital filters ######################

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype = "high", analog = False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def filter(x, sr=48000, lc=None, hc=None, order = 5, display = True):
    '''
    Filter signal with low-pass/high-pass/band-pass Butterworth filter.

    Parameters
    ----------
    x : input signal
    sr : sampling rate. 
        nyq = 0.5 * sr gets the frequency upper limit by the Nyquist Theorem.
    lc : low cutoff freq
    hc : high cutoff freq
    order : Butterworth order
    '''

    if lc is None and hc is None: # do nothing
        return x
    elif lc is None and hc > 0:
        b, a = butter_lowpass(hc, sr, order=order)
    elif hc is None and lc > 0:
        b, a = butter_highpass(lc, sr, order=order)
    elif lc is not None and hc is not None and lc > 0 and hc > 0:
        b, a = butter_bandpass(lc, hc, sr, order=order)
    else:
        b, a = 0, 0

    y = signal.lfilter(b, a, x)
    
    if display:

        # Plotting the frequency response.
        w, h = signal.freqz(b, a, worN=8000)
        plt.figure(figsize = (12, 12))
        plt.subplot(2, 1, 1)
        plt.plot(0.5*sr*w/np.pi, np.abs(h), 'b')
        if lc > 0:
            plt.plot(lc, 0.5*np.sqrt(2), 'ko')
            plt.axvline(lc, color='k')
        if hc > 0:
            plt.plot(hc, 0.5*np.sqrt(2), 'ko')
            plt.axvline(hc, color='k')
        plt.xlim(0, 0.5*sr)
        plt.title("Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(x, 'b-', label='data')
        plt.plot(y, 'g-', linewidth=2, label='filtered data')
        plt.xlabel('Time [sec]')
        plt.grid()
        plt.legend()

        plt.subplots_adjust(hspace=0.35)
        plt.show()

    return y

def filter_dataset(X, nlc=0.05, nhc=0.95):
    '''
    nlc : normalized low cutoff freq, a value from 0-1.0
    nhc : normalzied high cutoff freq, a value from 0-1.0
    '''
    Xf = []
    SR = 1000 # For Raman and MS, the x-axis is not time. So we use a FAKE SR 1000.
    NYC = SR * 0.5 # Nyquist Upperbound
    
    lc = None
    if nlc is not None and nlc > 0:
        lc = round(nlc*NYC)

    hc = None
    if nhc is not None and nhc > 0:
        hc = round(nhc*NYC)

    for x in X:
        y = filter(x, SR, lc, hc, display = False)        
        Xf.append(y)

    return np.array(Xf)

def diff_dataset(X):
    '''
    1st-order derivative
    '''
    # X_detrend = signal.detrend(X, axis=- 1, type='constant')
    X_diff = np.diff(X)
    return X_diff
    
def try_dataset_bdr(X, nlc_range = [0.002, 0.005, 0.01, 0.02]):
    '''
    Test a series of Butterworth low cutoff for baseline drift removal.
    '''

    for nlc in nlc_range:

        # Butterworth filter
        X_f = filter_dataset(X, nlc = nlc, nhc = None)  # axis = 0 for vertically; axis = 1 for horizontally
        plt.figure(figsize = (40,10))
        plt.plot(np.mean(X_f,axis=0).tolist()) 
        plt.title(u'Averaged Spectrum After Butterworth Highpass Filter. cutoff = ' + str(nlc), fontsize=30)
        plt.show()


########### Time Series Alignment Functions for e-nose/e-tongue ###############

def equalize_array_size(array1,array2):
    '''
    reduce the size of one sample to make them equal size. 
    The sides of the biggest signal are truncated
    Args:
        array1 (1d array/list): signal for example the reference
        array2 (1d array/list): signal for example the target
    Returns:
        array1 (1d array/list): middle of the signal if truncated
        array2 (1d array/list): middle of the initial signal if there is a size difference between the array 1 and 2
        dif_length (int): size diffence between the two original arrays 
    '''
    len1, len2 = len(array1), len(array2)
    dif_length = len1-len2
    if dif_length<0:
        array2 = array2[int(np.floor(-dif_length/2)):len2-int(np.ceil(-dif_length/2))]
    elif dif_length>0:
        array1 = array1[int(np.floor(dif_length/2)):len1-int(np.ceil(dif_length/2))]
    return array1,array2, dif_length

def chisqr_align(reference, target, roi=None, order=1, init=0.1, bound=1):
    '''
    Align a target signal to a reference signal within a region of interest (ROI)
    by minimizing the chi-squared between the two signals. Depending on the shape
    of your signals providing a highly constrained prior is necessary when using a
    gradient based optimization technique in order to avoid local solutions.
    Args:
        reference (1d array/list): signal that won't be shifted
        target (1d array/list): signal to be shifted to reference
        roi (tuple): region of interest to compute chi-squared
        order (int): order of spline interpolation for shifting target signal
        init (int):  initial guess to offset between the two signals
        bound (int): symmetric bounds for constraining the shift search around initial guess
    Returns:
        shift (float): offset between target and reference signal 
    
    Todo:
        * include uncertainties on spectra
        * update chi-squared metric for uncertainties
        * include loss function on chi-sqr
    '''
    reference, target, dif_length = equalize_array_size(reference,target)
    if roi==None: roi = [0,len(reference)-1] 
  
    # convert to int to avoid indexing issues
    ROI = slice(int(roi[0]), int(roi[1]), 1)

    # normalize ref within ROI
    reference = reference/np.mean(reference[ROI])

    # define objective function: returns the array to be minimized
    def fcn2min(x):
        shifted = shift(target,x,order=order)
        shifted = shifted/np.mean(shifted[ROI])
        return np.sum( ((reference - shifted)**2 )[ROI] )

    # set up bounds for pos/neg shifts
    minb = min( [(init-bound),(init+bound)] )
    maxb = max( [(init-bound),(init+bound)] )

    # minimize chi-squared between the two signals 
    result = minimize(fcn2min,init,method='L-BFGS-B',bounds=[ (minb,maxb) ])

    return result.x[0] + int(np.floor(dif_length/2))


def phase_align(reference, target, roi, res=100):
    '''
    Cross-correlate data within region of interest at a precision of 1./res
    if data is cross-correlated at native resolution (i.e. res=1) this function
    can only achieve integer precision 
    Args:
        reference (1d array/list): signal that won't be shifted
        target (1d array/list): signal to be shifted to reference
        roi (tuple): region of interest to compute chi-squared
        res (int): factor to increase resolution of data via linear interpolation
    
    Returns:
        shift (float): offset between target and reference signal 
    '''
    # convert to int to avoid indexing issues
    ROI = slice(int(roi[0]), int(roi[1]), 1)

    # interpolate data onto a higher resolution grid 
    x,r1 = highres(reference[ROI],kind='linear',res=res)
    x,r2 = highres(target[ROI],kind='linear',res=res)

    # subtract mean
    r1 -= r1.mean()
    r2 -= r2.mean()

    # compute cross covariance 
    cc = ccovf(r1,r2,demean=False,unbiased=False)

    # determine if shift if positive/negative 
    if np.argmax(cc) == 0:
        cc = ccovf(r2,r1,demean=False,unbiased=False)
        mod = -1
    else:
        mod = 1

    # often found this method to be more accurate then the way below
    return np.argmax(cc)*mod*(1./res)

    # interpolate data onto a higher resolution grid 
    x,r1 = highres(reference[ROI],kind='linear',res=res)
    x,r2 = highres(target[ROI],kind='linear',res=res)

    # subtract off mean 
    r1 -= r1.mean()
    r1 -= r2.mean()

    # compute the phase-only correlation function
    product = np.fft.fft(r1) * np.fft.fft(r2).conj()
    cc = np.fft.fftshift(np.fft.ifft(product))

    # manipulate the output from np.fft
    l = reference[ROI].shape[0]
    shifts = np.linspace(-0.5*l,0.5*l,l*res)

    # plt.plot(shifts,cc,'k-'); plt.show()
    return shifts[np.argmax(cc.real)]


def highres(y, kind='cubic', res=100):
    '''
    Interpolate data onto a higher resolution grid by a factor of *res*
    Args:
        y (1d array/list): signal to be interpolated
        kind (str): order of interpolation (see docs for scipy.interpolate.interp1d)
        res (int): factor to increase resolution of data via linear interpolation
    
    Returns:
        shift (float): offset between target and reference signal 
    '''
    y = np.array(y)
    x = np.arange(0, y.shape[0])
    f = interp1d(x, y,kind='cubic')
    xnew = np.linspace(0, x.shape[0]-1, x.shape[0]*res)
    ynew = f(xnew)
    return xnew,ynew


def align_nch_dataset(X, start, length, method = 'peak', display = True):
    '''
    Signal Alignment

    由于每个数据包含14个sensor channel，我们首先获取所有通道的均值波形。然后用均值波形进行对齐。
    Get averaged signal of all sensor channels. We use the averaged signal for alignment.

    start, length : cut [start, start+length) the shifted signal as the new aligned signal. 
        length should be the minimum common length of all samples.

    method : 
        'chisq' : 1. Alignment via chi-squared minimization. 对于SHIFT较大的信号，align效果不好
        This function shifts a signal by resampling it via linear interpolation, the edges of the signal are ignored because the algorithm focuses on aligning two signals within a region of interest. The shifted signal is then compared to a "template" signal where a chi-squared minimization occurs to derive the optimal offset between the two. Each signal (template and shifted) is normalized (mean=1) prior to shifting so that the algorithm focuses mainly on comparing the shape of the data. One caveat about this techqnique, it is prone to finding local minimum unless priors are heavily constrained since it uses a gradient-based optimization method.
        'phase' : 2. Alignment via Fourier space. 实测效果优于chisq
        Uses a phase correlation in fourier space (https://en.wikipedia.org/wiki/Phase_correlation). It applies the same technique as that link except in 1D instead of 2D. After independent testing this algorithm behaves in a similar manner as the chi-squared minimization except at a lower precision. The only caveat about the phase correlation is if you cross-correlate your data at the native resolution you can only achieve integer precision. To achieve sub-pixel/ sub-integer precision, I've added an option to upscale the data prior to the cross-correlation. By upscaling the data N times you can achieve a precision of 1/N as far as the shift value goes. That means if N=100, then the smallest shift value you can get is 0.01.
        Compares the shift between two signals that happen to be astrophysical spectra at two different wavelength regions.
        'peak' : 3. Alignment via the Highest Peak. 虽然最简单，但在实测数据集上效果比前两种好。也不需要前两种方法的超参数调优（ROI, etc.）
        Method 1 and 2 are provided by https://github.com/pearsonkyle/Signal-Alignment
    
    return:
        The aligned dataset, SHIFTS    
    '''

    # get ch-averaged data

    ch_averaged_samples = []

    for x in X:
        
        ch_averaged = []

        for xx in x: 
            if isinstance(xx, str): # if it is still in the comma-separated string format
                ch_averaged.append ( np.array(xx.split(",")).astype(np.float) ) 
            else: # already converted to array
                ch_averaged.append(xx)
        
        ch_averaged = np.array(ch_averaged).mean(axis = 0)
        ch_averaged_samples.append(ch_averaged)


    # Align signals

    L = len(ch_averaged_samples[0])
    MAX_IX = ch_averaged_samples[0].argmax() # highest peak index of 1st sample
    SHIFTS = [0] # use 1st sample as reference

    # align each signal with the first sample at ROI [0.25, 0.75]
    for idx, sample in enumerate(ch_averaged_samples):
        
        if (idx == 0):
            continue
        
        # align the shifted spectrum back to the real
        
        if method == 'phase':        
            s = phase_align(ch_averaged_samples[0], sample, roi = [round(L/4), round(L/4*3)]) # roi (tuple): region of interest to compute chi-squared
        elif method == 'chisq':
            s = chisqr_align(ch_averaged_samples[0], sample, roi = [round(L/2), round(L/4*3)]) # roi (tuple): region of interest to compute chi-squared
        else:
            MAX_IX2 = sample.argmax() 
            # align the shifted spectrum back to the real
            s = MAX_IX - MAX_IX2

        SHIFTS.append(s)

        if display:
            plt.plot(ch_averaged_samples[0],label='sample 0')
            plt.plot(sample + 0.1, label='sample ' + str(idx))
            plt.plot(shift(sample,s,mode='nearest') + 0.2,ls='--',label='aligned') 
            plt.legend(loc='best')
            plt.title('shift =' + str(round(s,3)))
            plt.show()

    # 截取生成对齐数据

    XSH = []

    for sidx, sample in enumerate(X):
        
        Xchs = []            
            
        for xx in x:
            if isinstance(xx, str): # if it is still in the comma-separated string format
                padded = list(np.array(xx.split(",")).astype(np.float))+[0]*length
            else: # already converted to array
                padded = list(xx) + [0]*length

            if start < SHIFTS[sidx]: # invalid data, set as 0-array
                Xchs.append([0]*length)
            else:
                Xchs.append( padded[start - SHIFTS[sidx] : ][:length] )
                
        XSH.append(np.array(Xchs))

    return np.array(XSH), SHIFTS

class BaselineDetector:
    '''
    Baseline detection using asymmetric least squares smoothing
    '''
    def __init__(self, lam = 1e8, p = 1e-3):
        self.lam = lam  # 1e2 - 1e9
        self.p = p  # 1e-3 - 1e-1

    @staticmethod
    def baseline_als(y, lam, p, niter=10):
        L = len(y)
        D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        z = 0
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z

    def detect_baseline(self, data):
        return np.apply_along_axis(lambda x: self.baseline_als(x, self.lam, self.p), 0, data)
    

def x_baseline_removal(X, lam = 1e8, p = 1e-3):
    '''
    Perform baseline removal on each sample in X.
    Caution: it takes long time for high-dimensional spectroscopic data.

    Parameters
    ----------
    X: 2D array, each row is a sample
    lam: float, regularization strength for 2nd-order derivative penalty
    p: float, p controls asymmetry of positive and negative residual penalties
    '''
    NX = []
    detector = BaselineDetector(lam = lam, p = p)
    for x in X:
        baseline = detector.detect_baseline(x)
        NX.append(x - baseline)
    return np.array(NX)

###################### dataset split algorithms #######################
# Source: https://hxhc.xyz/post/kennardstone-spxy/

def kennardstone_split(spectra, test_size=0.3, metric='euclidean', *args, **kwargs):
    """Kennard Stone Sample Split method
    Parameters
    ----------
    spectra: ndarray, shape of i x j
        i spectrums and j variables (wavelength/wavenumber/ramam shift and so on)
    test_size : float, int
        if float, then round(i x (1-test_size)) spectrums are selected as test data, by default 0.25
        if int, then test_size is directly used as test data size
    metric : str, optional
        The distance metric to use, by default 'euclidean'
        See scipy.spatial.distance.cdist for more infomation
    Returns
    -------
    select_pts: list
        index of selected spetrums as train data, index is zero based
    remaining_pts: list
        index of remaining spectrums as test data, index is zero based
    References
    --------
    Kennard, R. W., & Stone, L. A. (1969). Computer aided design of experiments.
    Technometrics, 11(1), 137-148. (https://www.jstor.org/stable/1266770)
    """

    if test_size < 1:
        train_size = round(spectra.shape[0] * (1 - test_size))
    else:
        train_size = spectra.shape[0] - round(test_size)

    if train_size > 2:
        distance = cdist(spectra, spectra, metric=metric, *args, **kwargs)
        select_pts, remaining_pts = max_min_distance_split(distance, train_size)
    else:
        raise ValueError("train sample size should be at least 2")

    return select_pts, remaining_pts


def stratified_kennardstone_split(X, y, test_size=0.3, metric='euclidean'):
    '''
    Perform a KS split per class to get a stratified training set and test set according to y.
    '''
    X_train = np.empty((0,X.shape[1]))
    X_test = np.empty((0,X.shape[1]))
    y_train = []
    y_test = []
    for yi in set(y):
        Xi = X[y == yi]
        Xi_train, Xi_test = kennardstone_split(Xi, test_size, metric)
        X_train = np.vstack((X_train, Xi[Xi_train,:]))
        X_test = np.vstack((X_test, Xi[Xi_test,:]))
        y_train += [yi] *len(Xi_train)
        y_test += [yi] * len(Xi_test)
    return X_train, X_test, y_train, y_test


def spxy_split(spectra, yvalues, test_size=0.3, metric='euclidean', *args, **kwargs):
    """SPXY Sample Split method
    Parameters
    ----------
    spectra: ndarray, shape of i x j
        i spectrums and j variables (wavelength/wavenumber/ramam shift and so on)
    test_size : float, int
        if float, then round(i x (1-test_size)) spectrums are selected as test data, by default 0.25
        if int, then test_size is directly used as test data size
    metric : str, optional
        The distance metric to use, by default 'euclidean'
        See scipy.spatial.distance.cdist for more infomation
    Returns
    -------
    select_pts: list
        index of selected spetrums as train data, index is zero based
    remaining_pts: list
        index of remaining spectrums as test data, index is zero based
    References
    ---------
    Galvao et al. (2005). A method for calibration and validation subset partitioning.
    Talanta, 67(4), 736-740. (https://www.sciencedirect.com/science/article/pii/S003991400500192X)
    """

    if test_size < 1:
        train_size = round(spectra.shape[0] * (1 - test_size))
    else:
        train_size = spectra.shape[0] - round(test_size)

    if train_size > 2:
        yvalues = yvalues.reshape(yvalues.shape[0], -1)
        distance_spectra = cdist(spectra, spectra, metric=metric, *args, **kwargs)
        distance_y = cdist(yvalues, yvalues, metric=metric, *args, **kwargs)
        distance_spectra = distance_spectra / distance_spectra.max()
        distance_y = distance_y / distance_y.max()

        distance = distance_spectra + distance_y
        select_pts, remaining_pts = max_min_distance_split(distance, train_size)
    else:
        raise ValueError("train sample size should be at least 2")

    return select_pts, remaining_pts


def max_min_distance_split(distance, train_size):
    """sample set split method based on maximun minimun distance, which is the core of Kennard Stone
    method
    Parameters
    ----------
    distance : distance matrix
        semi-positive real symmetric matrix of a certain distance metric
    train_size : train data sample size
        should be greater than 2
    Returns
    -------
    select_pts: list
        index of selected spetrums as train data, index is zero-based
    remaining_pts: list
        index of remaining spectrums as test data, index is zero-based
    """

    select_pts = []
    remaining_pts = [x for x in range(distance.shape[0])]

    # first select 2 farthest points
    first_2pts = np.unravel_index(np.argmax(distance), distance.shape)
    select_pts.append(first_2pts[0])
    select_pts.append(first_2pts[1])

    # remove the first 2 points from the remaining list
    remaining_pts.remove(first_2pts[0])
    remaining_pts.remove(first_2pts[1])

    for i in range(train_size - 2):
        # find the maximum minimum distance
        select_distance = distance[select_pts, :]
        min_distance = select_distance[:, remaining_pts]
        min_distance = np.min(min_distance, axis=0)
        max_min_distance = np.max(min_distance)

        # select the first point (in case that several distances are the same, choose the first one)
        points = np.argwhere(select_distance == max_min_distance)[:, 1].tolist()
        for point in points:
            if point in select_pts:
                pass
            else:
                select_pts.append(point)
                remaining_pts.remove(point)
                break
    return select_pts, remaining_pts



###################### end of dataset split algorithms ################