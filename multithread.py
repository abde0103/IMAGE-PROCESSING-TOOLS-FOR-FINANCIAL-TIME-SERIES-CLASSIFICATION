import yfinance as yf 
import multiprocessing as mp
from make_data import make_data
from pipeline import *



SP500 = yf.download('SPY', keepna = True)
data = make_data(SP500)
signal = data['Adj Close'].to_numpy()
kwargs = {'b' : 2., 'c' : 1. }

window_size = 360
level_discrete_wavelet_transform = 2






if __name__ == '__main__':
    jobs =[]
    n = signal.shape[0]
    threads = mp.cpu_count()
    
    for i in range(threads):
        
        if (i == threads - 1):
            thread =mp.Process(
                target=generate_NN_dataset,
                args = (
                signal[i*n//threads:] ,
                window_size,
                i,
                'mean',
                1,
                2,
                'db4',
                'cmor'),
                kwargs = kwargs        
            
        )
            thread.daemon = True 
            thread.start()
                
        
        else:
  
            thread = mp.Process(
                target=generate_NN_dataset,
                args = (
                signal[i*n//threads:(i+1)*n//threads] ,
                window_size,
                i,
                'mean',
                1,
                2,
                'db4',
                'cmor'),
                kwargs = kwargs)
            
            thread.daemon = True 
            thread.start()
        
        jobs.append(thread)
    
    for i in jobs :
        i.join()
    for i in jobs :
        i.close()