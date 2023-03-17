import matplotlib.pyplot as plt
import numpy as np 
try:
    from numpy.lib.stride_tricks import  sliding_window_view  # New in version 1.20.0

    def get_trajectory_matrix(arr, window_shape, jump=1):
        return sliding_window_view(x=arr, window_shape=window_shape)[::jump]
    
except ImportError:
    def get_trajectory_matrix(arr, window_shape, jump=1):
        n_rows = ((arr.size - window_shape) // jump) + 1
        n = arr.strides[0]
        return np.lib.stride_tricks.as_strided(
            arr, shape=(n_rows, window_shape), strides=(jump * n, n)
        )
    


def average_adiag(x):
    """Average antidiagonal elements of a 2d array
    Parameters:
    -----------
    x : np.array
        2d numpy array of size

    Return:
    -------
    x1d : np.array
        1d numpy array representing averaged antediangonal elements of x

    """
    x1d = [np.mean(x[::-1, :].diagonal(i)) for i in
           range(-x.shape[0] + 1, x.shape[1])]
    return np.array(x1d)


def SSA (sample , window_SSA = 10, thresh = 0.91, show = True):
    """
    Generates denoised signal based on SSA
    
    Parameters
    ----------
    sample : array
        portion of the signal
        
    window_SSA : int
        length of fragments in the Trajectory matrix
        
    Thresh : float between 0 and 1
        The proportion of explained variance : how to choose the number of eignevalues to take
        
    show : boolean
        Whether to return plots
        
    Return
    -------
    array:
        Denoised sample 
    """

    ## Normalize sample (eignevalues will have more meaning )
    sample_mean, sample_std  = sample.mean(), sample.std()
    normalized_sample = (sample - sample_mean)/sample_std
    trajectory_matrix = get_trajectory_matrix(normalized_sample, window_SSA)
    
    # SVD
    u, eigenvals, vh = np.linalg.svd(trajectory_matrix, full_matrices=False)
    
    
    ## SSA decomposition
    
    ssa_decomposition = np.zeros((sample.size, window_SSA))
    sum_eigen_vals = eigenvals.sum() 
    current_sum_eigenvalues = 0
    
    for (ind, (left, sigma, right)) in enumerate(zip(u.T, eigenvals, vh)):
        
        ssa_decomposition.T[ind] = average_adiag(sigma * np.dot(left.reshape(-1, 1), right.reshape(1, -1)))
        current_sum_eigenvalues += sigma/sum_eigen_vals
        if(current_sum_eigenvalues>= thresh):
            if (show):
                print ('number of eigenvalues taken is' , str(ind+ 1 ))
            
            break
    
    
    
    ### Generate image of contribution of each component
    if (show): 
        fig, ax_arr = plt.subplots(
        nrows=window_SSA // 3 + 1,
        ncols=3,
        figsize=(20, 3 * (window_SSA // 3 + 1)),
    )

        for (ind, (component, ax)) in enumerate(
        zip(ssa_decomposition.T, ax_arr.flatten())
        ):
            ax.plot(component)
            ax.set_xlim(0, component.size)
            ax.set_title(f"Component n°{ind}")

            fig.savefig('SSA_reconstruction_components.png')
        plt.figure()
        plt.bar (range(len(eigenvals)) , eigenvals)
        plt.title('Eigenvalues')
        
                 
    return ssa_decomposition.sum(1)*sample_std + sample_mean

    