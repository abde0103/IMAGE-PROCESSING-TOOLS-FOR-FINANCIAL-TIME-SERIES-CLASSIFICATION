import numpy as np 
import os
from tqdm import tqdm
from DWT import discrete_wavelet_denoise, log_return 
from CWT import CWT
from SSA import *
from Short_time_fourier import STFourier




def generate_labels(signal, window_size, method ='mean' ):
    
    if (method == 'mean'):
        moving_average = np.convolve(signal, np.ones(window_size)/window_size, mode='valid')[:-1] ## exclude last sample cuz no label
        next_day = signal[window_size : ]
        labels = 1*(next_day>moving_average)
    
    return labels
   
    
def generate_spectrogram_dwt(sample ,path,level_discrete_wavelet_transform = 2,discrete_wavelet = 'db4') : 
    
    """
    generates a Fourier spectrogram (Short time Fourier) from a portion of the signal based on DWT denoisng as preprocessing
    
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
    
    Returns 
    --------
    generates a Fourier spectrogram from a raw signal
        
    """
   
    denoised_sample = discrete_wavelet_denoise (sample , wavelet = discrete_wavelet, level = level_discrete_wavelet_transform)
    log_ret = log_return(denoised_sample)
    STFourier (log_ret, path = path,show = False  ) 

    
    
def generate_scalogram_dwt(sample ,path,level_discrete_wavelet_transform = 2,discrete_wavelet = 'db4',continuous_wavelet = 'cmor', **kwargs ) : 
    """
    generates a scalogram from a portion of the signal based on DWT denoisng as preprocessing
    
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
    

def generate_spectrogram_ssa(sample ,path,  window_SSA = 5, thresh = 0.9) : 
    """
    generates a Fourier spectrogram (Short time Fourier) from a portion of the signal based on SSA denoisng as preprocessing
    
    Parameters 
    ----------
    sample : array
        the portion of the signal to be analyzed
    path : string
        path to save the image
    
    window_SSA : int
        length of fragments in the Trajectory matrix
        
    thresh : float between 0 and 1
        The proportion of explained variance : how to choose the number of eignevalues to take
    
    """
   
    denoised_sample = SSA (sample , window_SSA = window_SSA, thresh = thresh, show = False)
    log_ret = log_return(denoised_sample)
    STFourier (log_ret, path = path,show = False  )
    
    
    
def generate_scalogram_ssa(sample ,path,  window_SSA = 5, thresh = 0.9, continuous_wavelet = 'cmor', **kwargs ) : 
    """
    generates a scalogram from a portion of the signal based on SSA denoisng as preprocessing
    
    Parameters 
    ----------
    sample : array
        the portion of the signal to be analyzed
    path : string
        path to save the image
    
    window_SSA : int
        length of fragments in the Trajectory matrix
        
    thresh : float between 0 and 1
        The proportion of explained variance : how to choose the number of eignevalues to take
    
    continous_wavelet : string
        Type of the CWT (adapted to the package pywt)
        
    kwargs : dict
        Parameters of the continuous wavelet
        
    """
   
    denoised_sample = SSA (sample , window_SSA = window_SSA, thresh = thresh, show = False)
    log_ret = log_return(denoised_sample)
    CWT (log_ret,wavelet = continuous_wavelet, show = False ,path = path, **kwargs  )
    
    

def generate_NN_dataset(
    signal ,
    window_size,
    thread = 0,
    fourier= False,
    denoising_method = 'dwt',
    method_labels = 'mean',
    window_SSA = 5,
    thresh = 0.9,
    jump = 1,
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
    
    thread : int
        num of thread that executes this function: purely technical (forget about it)
        
    fourier : boolean
        whether to generatea Fourier spectrogram (True) or a continuous wavelet scalogram (False)
    
    denoising_method : string 
        dwt or ssa
        
    method_labels : string
        default : 'mean' 
        how to generate the labels : 'mean' means the labels are generated 
        as follows : ref the paper https://arxiv.org/pdf/2008.06042.pdf
     
     window_SSA : int
        length of fragments in the Trajectory matrix (case SSa denoising)
        
    thresh : float between 0 and 1
        The proportion of explained variance : how to choose the number of eignevalues to take (case SSa denoising)
    
    level_discrete_wavelet_transform :int
        level of discrete wavelet transform (case dwt denoising)
        
    discrete_wavelet : string
        Type of the DWT (adapted to the package pywt) (case dwt denoising)
    
    continous_wavelet : string
        Type of the CWT (adapted to the package pywt)
        
    kwargs : dict
        Parameters of the continuous wavelet
        
    
    Returns
    --------
    images in folders 0/ and 1/ according to the label of the image
    """
    if (denoising_method == 'dwt'):
        params = '_level' + str(level_discrete_wavelet_transform) + '_jump' + str(jump)
        
    if (denoising_method == 'ssa'):
        params = '_window_ssa' + str(window_SSA) + '_thresh' + str(thresh) + '_jump' + str(jump) 
    
    # create folder per label
    if (fourier):
        parent_folder = 'Fourrier_window_size' + str(window_size) + params
    else :
        parent_folder = 'Wavelet_window_size' + str(window_size) + params
        
    os.makedirs(os.path.join( parent_folder ,'0'), exist_ok=True)
    os.makedirs(os.path.join(parent_folder,'1'), exist_ok=True)
    
    #labels
    labels =  generate_labels(signal, window_size, method =method_labels )
    
    #create images
    for i in tqdm(range(0,signal.shape[0] - window_size + 1, jump)):
  
        sample = signal[i:i+window_size]
        path = os.path.join(parent_folder, os.path.join(str(labels[i]), 'thread_'+str(thread)+'_' + str(i)))
        
        
        ## If DWT
        if(denoising_method == 'dwt'):
            
            if (fourier):
                generate_spectrogram_dwt(
                    sample ,
                    path,
                    level_discrete_wavelet_transform,
                    discrete_wavelet)
            
            else : 
                generate_scalogram_dwt(
                    sample ,
                    path,
                    level_discrete_wavelet_transform,
                    discrete_wavelet,
                    continuous_wavelet,
                     **kwargs )
       
        ## If SSA
        if(denoising_method == 'ssa'):
            
            if (fourier):
                generate_spectrogram_ssa(
                    sample ,
                    path,
                    window_SSA,
                    thresh) 
            else: 
                generate_scalogram_ssa(
                    sample ,
                    path,
                    window_SSA,
                    thresh, 
                    continuous_wavelet ,
                    **kwargs ) 
 