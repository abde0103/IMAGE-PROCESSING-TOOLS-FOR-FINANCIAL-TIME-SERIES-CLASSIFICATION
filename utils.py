import pandas as pd 
import numpy as np 

# Simple Moving Average 
def SMA(data, ndays): 
    SMA_ = pd.Series(data['Adj Close'].rolling(ndays).mean(), name = 'SMA'+str(ndays)) 
    data = data.join(SMA_) 
    return data


# Exponentially-weighted Moving Average 
def EWMA(data, ndays): 
    EMA_ = pd.Series(data['Adj Close'].ewm(span = ndays, min_periods = ndays - 1).mean(), 
                 name = 'EWMA_' + str(ndays)) 
    data = data.join(EMA_) 
    return data

# Compute the Bollinger Bands 
def BBANDS(data, window):
    MA = data['Adj Close'].rolling(window).mean()
    SD = data['Adj Close'].rolling(window).std()
#     data['MiddleBand'] = MA ## Equal MA 30 
    data['UpperBand'] = MA + (2 * SD) 
    data['LowerBand'] = MA - (2 * SD)
    return data

# Returns RSI values
def rsi(data, periods = 14):
    
    close_delta = data['Adj Close'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()

    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    
    data['RSI'+str(periods)] = rsi
    return data



def gain(x):
    return ((x > 0) * x).sum()


def loss(x):
    return ((x < 0) * x).sum()


# Calculate money flow index
def mfi(data, n=14):
    
    high = data['Low']
    low  = data['High']
    close = data['Adj Close']
    volume = data['Volume']
    
    typical_price = (high + low + close)/3
    money_flow = typical_price * volume
    mf_sign = np.where(typical_price > typical_price.shift(1), 1, -1)
    signed_mf = money_flow * mf_sign
    mf_avg_gain = signed_mf.rolling(n).apply(gain, raw=True)
    mf_avg_loss = signed_mf.rolling(n).apply(loss, raw=True)
    
    
    result = (100 - (100 / (1 + (mf_avg_gain / abs(mf_avg_loss))))).to_numpy()
    data["MFI"+str(n)] = result
    
    return data


# Returns ATR values
def atr(data, n=14):
    high = data['Low']
    low  = data['High']
    close = data['Adj Close']
    tr = np.amax(np.vstack(((high - low).to_numpy(), (abs(high - close)).to_numpy(), (abs(low - close)).to_numpy())).T, axis=1)
    data['ATR'+str(n)] = pd.Series(tr).rolling(n).mean().to_numpy()
    return data


# Returns the Force Index 
def ForceIndex(data, ndays): 
    FI = pd.Series(data['Adj Close'].diff(ndays) * data['Volume'], name = 'ForceIndex'+str(ndays)) 
    data = data.join(FI) 
    return data

# Ease of Movement 
def EMV(data, ndays): 
    dm = ((data['High'] + data['Low'])/2) - ((data['High'].shift(1) + data['Low'].shift(1))/2)
    br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
    EMV = dm / br 
    EMV_MA = pd.Series(EMV.rolling(ndays).mean(), name = 'EMV'+ str(ndays)) 
    data = data.join(EMV_MA) 
    return data 