import yfinance as yf 
import multiprocessing as mp
from make_data import make_data
from pipeline import *
import argparse


SP500 = yf.download('SPY', keepna = True)
data = make_data(SP500)
signal = data['Adj Close'].to_numpy()
kwargs = {'b' : 2., 'c' : 1. }


# window_size = 14
# level_discrete_wavelet_transform = 2
# jump = 8



parser = argparse.ArgumentParser(description='Generation of dataset')
parser.add_argument('--w', type = int, help='window_size', default = 14 )
parser.add_argument('--level', type = int , help='level discrete wavelet', default = 2)
parser.add_argument('--jump', type = int, help='jump', default = 8 )
parser.add_argument('--method', type = str, help='denoising method : dwt ou ssa', default = 'dwt' )
parser.add_argument('--w_ssa', type = int, help='Window shape for SSA denoising', default = 5)
parser.add_argument('--thresh', type = float, help='Threshold SSA denoising', default = 0.9)



args = parser.parse_args()


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
                args.w,
                i,
                args.method,
                'mean',
                args.w_ssa,
                args.thresh,
                args.jump,
                args.level,
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
                args.w,
                i,
                args.method,
                'mean',
                args.w_ssa,
                args.thresh,
                args.jump,
                args.level,
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