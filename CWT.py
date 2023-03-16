import numpy as np
import pywt
import matplotlib.pyplot as plt 


def CWT (denoised_logreturn,show = False , wavelet = 'cmor' , path ='',  **kwargs ) : 
    
    """
    Return scalogram given logreturn as input
    
    Parameters
    ----------
    denoised_logreturn : array
        denoised log return of length n
    wavelet : string
        type of wavelet: same format as wavelet in 'pywt.cwt'
        default : Morlet
    
    kwargs : parameters of the wavelet  
    
    
    Returns
    -------
    Returns the values of the pixel of image
    Creates an PNG image
    
    """
    scales = np.arange(2**8)+1
    cwtmatr, freqs = pywt.cwt(denoised_logreturn, scales, 'cmor'+str(kwargs['b'])+'-'+str(kwargs['c']))
    
    # Plot
    
    plt.imshow(np.abs(cwtmatr), cmap = 'jet',  aspect = 'auto') 
    plt.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    plt.xlabel('time')
    plt.ylabel('scales')
    plt.yscale('log', base = 2)
    plt.ylim([256,1])
    
    plt.savefig(path)
    
    if show:
        plt.show() 
    
    return np.abs(cwtmatr)