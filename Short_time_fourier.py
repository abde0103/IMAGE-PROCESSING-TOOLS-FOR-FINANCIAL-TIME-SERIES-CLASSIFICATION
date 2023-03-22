import scipy
import numpy as np 
import matplotlib.pyplot as plt

def STFourier (denoised_logreturn, path ='',show = False  ) : 
    
    """
    Return spectrogram given logreturn as input
    
    Parameters
    ----------
    denoised_logreturn : array
        denoised log return of length n
    
    path : string
        Path tosave the image
    
    show : boolean
        whether to plot the figure in the STDOUT
    
    
    
    Returns
    -------
    Returns the values of the pixel of image
    Creates an PNG image
    
    """
    
    FS = 1
    window_size = denoised_logreturn.shape[0]
    f, t, Sxx = scipy.signal.spectrogram(denoised_logreturn, FS, nperseg = int(window_size//2) )
    
    
    plt.pcolormesh(t, f, Sxx, shading='gouraud', cmap = 'jet')
#     plt.axis ('off')
    plt.savefig(path)
    
    return np.abs(Sxx)