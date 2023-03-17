import minepy
from pipeline import generate_labels
from make_data import make_data 
import yfinance as yf
import argparse

### Parse arguments 
parser = argparse.ArgumentParser(description='Indicators selection')
parser.add_argument('--w', type = int, help='window_size', default = 360 )
args = parser.parse_args()



## Finance datasets
SP500 = yf.download('SPY', keepna = True )
data = make_data (SP500)



def select_features (data , n_select_features = 3):
    ### Labels generation 
    signal = data['Adj Close'].to_numpy()
    labels = generate_labels(signal , parser.w )

    ### MICs
    mics = minepy.cstats(data.iloc[args.w:,:].to_numpy().T ,  labels.reshape(1,-1))[0].squeeze()
    return data.iloc[:,mics.argsort()[-n_select_features:]], mics

