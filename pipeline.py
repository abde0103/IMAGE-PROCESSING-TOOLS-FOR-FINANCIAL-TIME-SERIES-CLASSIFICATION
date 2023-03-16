import numpy as np 
import os
from DWT import discrete_wavelet_denoise, log_return 
from CWT import CWT
from tqdm import tqdm


def generate_labels(signal, window_size, method ='mean' ):
    
    if (method == 'mean'):
        moving_average = np.convolve(signal, np.ones(window_size)/window_size, mode='valid')[:-1] ## exclude last sample cuz no label
        next_day = signal[window_size : ]
        labels = 1*(next_day>moving_average)
    
    return labels
        
    

def generate_image(sample ,path, level_discrete_wavelet_transform,discrete_wavelet = 'db4',continuous_wavelet = 'cmor', **kwargs ) : 
    """
    generates an image from a portion of the signal
    
    Parameters 
    ----------
    sample : array
        the portion of the signal to be analyzed
    path : string
        path to save the image
    
    level_discrete_wavelet_transform :int
        level of discrete wavelet transform
        
    discrete_wavelet : string
        Type of the DWT (adapted to the package pywt)
    
    continous_wavelet : string
        Type of the CWT (adapted to the package pywt)
        
    kwargs : dict
        Parameters of the continuous wavelet
        
    """
    denoised_sample = discrete_wavelet_denoise (sample , wavelet = discrete_wavelet, level = level_discrete_wavelet_transform)
    log_ret = log_return(denoised_sample)
    CWT (log_ret,wavelet = continuous_wavelet, show = False ,path = path, **kwargs  )
    
    

def generate_NN_dataset(
    signal ,
    window_size,
    method_labels = 'mean',
    level_discrete_wavelet_transform = 2,
    discrete_wavelet = 'db4',
    continuous_wavelet = 'cmor', 
    **kwargs
    ) :
    
    """
    generates two folders of images according to the label
    
    Parameters 
    ----------
    signal : array
        the whole signal to be analyzed (one time series)
        
    window_size : int
        The length of the generate photo
        
    method_labels : string
        default : 'mean' 
        how to generate the labels : 'mean' means the labels are generated 
        as follows : ref the paper https://arxiv.org/pdf/2008.06042.pdf
    
    level_discrete_wavelet_transform :int
        level of discrete wavelet transform
        
    discrete_wavelet : string
        Type of the DWT (adapted to the package pywt)
    
    continous_wavelet : string
        Type of the CWT (adapted to the package pywt)
        
    kwargs : dict
        Parameters of the continuous wavelet
        
    
    Returns
    --------
    images in folders 0/ and 1/ according to the label of the image
    """
    
    # create folder per label
    os.makedirs(os.path.join(str(window_size),'0'), exist_ok=True)
    os.makedirs(os.path.join(str(window_size),'1'), exist_ok=True)
    
    #labels
    labels =  generate_labels(signal, window_size, method =method_labels )
    
    #create images
    for i in tqdm(range(signal.shape[0] - window_size + 1)):
  
        sample = signal[i:i+window_size]
      
        path = os.path.join(str(window_size), os.path.join(str(labels[i]), str(i)))
        
        generate_image(
            sample ,
            path,
            level_discrete_wavelet_transform,
            discrete_wavelet,
            continuous_wavelet,
             **kwargs ) 