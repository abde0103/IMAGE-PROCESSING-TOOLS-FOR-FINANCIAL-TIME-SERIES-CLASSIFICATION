import argparse
from tqdm import tqdm
import os
import PIL.Image as Image
import torch
from data import get_test_data_loader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from cnn import Cnn

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--data', type=str, default='test_data', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--out_dir', type=str, default='test_results', metavar='D',
                    help="name of the output csv file")
parser.add_argument('--in_channel', type= int, default=1, help="number of input channel")
parser.add_argument('--out_channel1', type= int, default=10, help="number of output channel for the first conv layer")
parser.add_argument("--resizing", type=int,default=50,help='location to store the trained')


args = parser.parse_args()
use_cuda = torch.cuda.is_available()
if use_cuda:
    print('Using GPU')
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')






def test(dataloader,model, device):

    model.eval()
    predictions = []
    labels = []
    for images, label in dataloader:
        images = images.to(device)
        output = F.softmax(model(images),dim=1)
        output = output.detach().cpu().numpy()
        predictions.append(output)
        labels.append(label.numpy())
    
    predictions = np.concatenate(predictions,axis=0)
    labels = np.hstack(labels)
    return predictions, labels

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
        
if __name__=="__main__":
    #Dataloader
    dataloader,classes = get_test_data_loader(args.data,32,shuffle=False,resize=args.resizing)

    state_dict = torch.load(args.model)
    resize = args.resizing
    k_size = 3
    linear_size = 20*(((resize - k_size + 1)//2 - k_size +1)//2)**2
    model = Cnn(len(classes),in1=args.in_channel, out1=args.out_channel1, linear_size=linear_size)
    model.load_state_dict(state_dict)
    model.to(device)
    pred,true_lab = test(dataloader,model,device)
    pred_lab = np.argmax(pred,axis=1)

    f1 = f1_score(true_lab,pred_lab)
    acc = accuracy_score(true_lab,pred_lab)
    recall = recall_score(true_lab,pred_lab)
    metric = [f"F1 score : {f1}\n",f"Accuracy : {acc}\n",f"recall : {recall}\n"]
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.out_dir+'/metric.txt','w') as f:
        f.writelines(metric)
    
    fpr,tpr,tr = roc_curve(true_lab,pred[:,1])
    x = np.linspace(0,1,len(fpr))
    plt.figure()
    plt.plot(fpr,tpr, label='our model')
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.plot(x,x, label = 'random')
    plt.legend(loc ="best")
    plt.savefig(args.out_dir+'/roc_curve.png')

    plt.figure()
    conf = confusion_matrix(true_lab,pred_lab,labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=conf,display_labels=[0,1])
    disp.plot()
    plt.savefig(args.out_dir+'/confusion_matrix.png')

