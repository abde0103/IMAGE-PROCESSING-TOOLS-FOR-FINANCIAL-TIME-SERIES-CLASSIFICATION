import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt

def log_return (X) : 
    differences = np.diff(X, prepend = np.nan)
    ret =  np.log1p(differences / X)
    ret[0] = 0
    return ret


def ValSUREThresh(X):
    """
    Adaptive Threshold Selection Using Principle of SURE

    Parameters
    ----------
    X: array
         Noisy Data with Std. Deviation = 1

    Returns
    -------
    tresh: float
         Value of Threshold

    """
    n = np.size(X)
    a = np.sort(np.abs(X))**2

    c = np.linspace(n-1,0,n)
    s = np.cumsum(a)+c*a # np.flip(a)
    risk = (n - (2 * np.arange(n)) + s)/n
    ibest = np.argmin(risk)
    THR = np.sqrt(a[ibest])
    return THR

def discrete_wavelet_denoise (data , wavelet = 'db4', level = 5):
    
    """
    Reconstruct signal after wavelet decompostion and soft thresholding using Rigrsure threshold
    
    Parameters
    ----------
    data : array
        noisy data length n
        
    wavelet : string 
        type of wavelet
    
    level : int
        number of levels 
        
    
    Returns
    -------
    array:
        reconstructed signal 
    """
    

    ## normalization trick
    std_data = 1# np.std(data)
    normalized_data = data/std_data
    
    ## Wavelet decomposition 
    wl = pywt.Wavelet(wavelet)
    coeff_all = pywt.wavedec(normalized_data, wl, level=level) #coef_all[0] : approximation coef of last level, coef_all[1:] : detail coef from last level to 1st level
    

    
    ## Threshold

    for i,detail in enumerate(coeff_all):
        
        if i>0:                      ## avoid approximation coef 
            
            detail_std = np.std(detail)
            normalized_detail = detail/detail_std
            coeff_all[i] =detail_std*pywt.threshold(normalized_detail, value=ValSUREThresh(normalized_detail) , mode='soft')
    

    
    
    ## Setting detail of last level to 0
    coeff_all[1][:] = 0
    
    ## reconstruct
    recon = pywt.waverec(coeff_all, wavelet= wl)
    
    return recon*std_data

    
    ## Example 
try:
    data = pd.read_csv('example.csv')
    k = data.shape[0]
    plt.plot(data['Adj Close'].to_numpy()[:k], label = 'original')
    plt.plot(discrete_wavelet_denoise (data['Adj Close'][:k]), label = 'denoised')
    plt.legend()
    plt.savefig('example.png')
    plt.show()
    print ('check the example.png created by this file')
except FileNotFoundError:
    print('File example.csv does not exist. Run make_data file before this file')
#     raise Exception()
    

 
 