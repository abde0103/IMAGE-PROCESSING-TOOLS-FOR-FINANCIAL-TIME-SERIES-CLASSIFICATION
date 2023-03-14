import yfinance as yf 
from utils import *

def make_data(SP500):
    
    n_ma = [5,7,14,30,60,90]
    data = SP500.copy()

    ## moving average
    for n in n_ma : 
        data = SMA(data,n)

    ## Exponential moving average
    for n in n_ma : 
        data = EWMA(data,n)   
    
    ## BBands
    data = BBANDS(data, window= 30)

    ## RSi 
    for n in n_ma :
        data = rsi(data,n)
    
    ## MFI
    for n in n_ma :
        data = mfi(data,n)
    
    ## ATR 
    for n in n_ma :
        data = atr(data,n)
    
    ## Force index
    for n in n_ma :
        data = ForceIndex(data,n)

    ## EMV
    for n in n_ma :
        data = EMV(data,n)
    
    return data


# Example 
SP500 = yf.download('SPY', keepna = True )
data = make_data(SP500)
data.to_csv('example.csv')