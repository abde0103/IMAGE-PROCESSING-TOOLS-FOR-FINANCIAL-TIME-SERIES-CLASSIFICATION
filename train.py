
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from tqdm import tqdm
from cnn import Cnn
import numpy as np
from data import get_train_data_loader,get_test_data_loader,get_eval_data_loader,plot_imageTensor

parser = argparse.ArgumentParser(description='training and cross validation')
parser.add_argument('--optim', type = str, help='select between sgd or adam', default = 'sgd' )
parser.add_argument('--weight_decay', type = float , default = 0)

parser.add_argument('--data', type=str, default='example', metavar='D',
                    help="folder where data is located. train_images/ and test_images/ need to be found in the folder")
parser.add_argument('--batch_size', type=int, default=32, metavar='B',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--eval', action='store_true',
                    help='folder where experiment outputs are located.')
parser.add_argument('--cv', type = int, default=1 ,help='number of fold for cross validation')
parser.add_argument('--test_path', type=str,  metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--in_channel', type= int, default=1, help="number of input channel")
parser.add_argument('--out_channel1', type= int, default=10, help="number of output channel for the first conv layer")
parser.add_argument("--save_model", type=str,help='location to store the trained')
parser.add_argument("--resizing", type=int,default=50,help='location to store the trained')
parser.add_argument("--kernel", type=int,default=3,help='kernel windows size')

# Data initialization and loading

def train_epoch(dataloader,model,optimizer,criterion, device,W = None):
    train_loss,train_correct=0.0,0
    model.train()
    for images, labels in dataloader:
        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()
        #print(f"loss {loss.data.item()}")
    
    return train_loss,train_correct

def valid_epoch(dataloader,model,optimizer,criterion, device):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for images, labels in dataloader:
        images,labels = images.to(device),labels.to(device)
        output = model(images)
        loss=criterion(output,labels)
        valid_loss+=loss.item()*images.size(0)
        scores, predictions = torch.max(output.data,1)
        val_correct+=(predictions == labels).sum().item()
    return valid_loss,val_correct

def train(dataloader,model,optimizer,criterion, device, eval,epochs, cv = False):
    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
    
    if eval:
        test_loader = dataloader['val'][0]
    for epoch in range(epochs):
        #print(f"epoch {epoch}/{epochs}")
        train_loss, train_correct=train_epoch(train_loader,model,optimizer,criterion,device,W)
        if eval:
            test_loss, test_correct=valid_epoch(test_loader,model,optimizer,criterion,device)

        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_correct / len(train_loader.sampler) * 100
        if eval:
            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100
            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                             epochs,
                                                                                                             train_loss,
                                                                                                             test_loss,
                                                                                                             train_acc,
                                                                                                             test_acc))
        else:
            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Training Acc {:.2f} %".format(epoch + 1,
                                                                                                             epochs,
                                                                                                             train_loss,
                                                                                                             train_acc))
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        if eval:
            history['test_acc'].append(test_acc)
            history['test_loss'].append(test_loss)

        if args.save_model and (not eval or args.test_path):
            os.makedirs(args.save_model, exist_ok=True)
            model_file =args.save_model +'/model_'+'_epoch_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), model_file)
 
    # avg_train_loss = np.mean(history['train_loss'])
    # avg_train_acc = np.mean(history['train_acc'])
    # if eval:
    #     avg_test_loss = np.mean(history['test_loss'])
    #     avg_test_acc = np.mean(history['test_acc'])
    #     print("Average Training Loss: {:.4f} \t Average Test Loss: {:.4f} \t Average Training Acc: {:.3f} \t Average Test Acc: {:.3f}".format(avg_train_loss,avg_test_loss,avg_train_acc,avg_test_acc))
    # else:
    #     print("Average Training Loss: {:.4f}  \t Average Training Acc: {:.3f} ".format(avg_train_loss,avg_train_acc))

    if cv:
        history['train_loss'] = history['train_loss'][-1]
        history['train_acc'] = history['train_acc'][-1]
        history['test_acc'] = history['test_acc'][-1]
        history['test_loss'] = history['test_loss'][-1]

    
    return history


args = parser.parse_args()

if __name__ == "__main__":
    resize = args.resizing
    k_size = args.kernel
    linear_size = 20*(((resize - k_size + 1)//2 - k_size +1)//2)**2
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.eval:
        print("training and validating")
        if args.cv <= 1:
            data_loader,classes= get_eval_data_loader(args.data, args.batch_size,resize=resize)
        else:
            print(f"Cross validation with cv {args.cv}")
            data_loaders,classes= get_eval_data_loader(args.data, args.batch_size,cv=args.cv,resize=resize)
    elif args.test_path:
        print("training and testing")
        test_loader = get_test_data_loader(args.test_path, args.batch_size,resize=resize)
        train_loader = get_train_data_loader(args.data,args.batch_size,resize=resize)
        data_loader = {'train':[train_loader],'val':[test_loader]}
    else:
        data_loader,classes =get_train_data_loader(args.data,args.batch_size,resize=resize)
    
    assert len(classes) == 2, "the number of classes must be equals to 2"
    model = Cnn(len(classes),in1=args.in_channel, out1=args.out_channel1, linear_size=linear_size)
    if use_cuda:
        print('Using GPU')
        model.cuda()
        device = torch.device('cuda')
    else:
        print('Using CPU')
        device = torch.device('cpu')
    
    
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(),lr = args.lr, weight_decay= args.weight_decay)
    
    if args.eval and args.cv >1:
        history = {'train_loss':[],'train_acc':[],'test_acc':[], 'test_loss':[]}
        for i in range(args.cv) :
            print(f"Fold {i+1}/{args.cv}...")
            model = Cnn(len(classes),in1=args.in_channel, out1=args.out_channel1, linear_size=linear_size).to(device)
            if args.optim == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            elif args.optim == 'adam':
                optimizer = optim.Adam(model.parameters(),lr = args.lr, weight_decay= args.weight_decay)
            train_loader = data_loaders['train'][0]
            targets = np.array(data_loaders['train'][i].dataset.targets)
            W = np.zeros(2)
            W[0] = len(targets)/np.sum(targets == 0)
            W[1] = len(targets)/np.sum(targets == 1)
            W = torch.Tensor(W).to(device=device)
            W = torch.log(torch.log(W) + 1)+1
            criterion = nn.CrossEntropyLoss(reduction='mean',weight=W)
            temp = train({'train':[data_loaders['train'][i]],'val':[data_loaders['val'][i]]},\
                  model,optimizer,criterion,device,args.eval or args.test_path,args.epochs, cv=True)
            history['train_loss'].append(temp['train_loss'])
            history['train_acc'].append(temp['train_acc'])
            history['test_loss'].append(temp['test_loss'])
            history['test_acc'].append(temp['test_acc'])
        avg_train_loss = np.mean(history['train_loss'])
        avg_train_acc = np.mean(history['train_acc'])
        avg_test_loss = np.mean(history['test_loss'])
        avg_test_acc = np.mean(history['test_acc'])
        print("\n")    
        print("Average Training CV Loss: {:.4f} \t Average Test CV Loss: {:.4f} \t Average Training CV Acc: {:.3f} \t Average Test CV Acc: {:.3f}".format(avg_train_loss,avg_test_loss,avg_train_acc,avg_test_acc))

        
    else:
        train_loader = data_loader['train'][0]
        targets = np.array(train_loader.dataset.targets)
        W = np.zeros(2)
        W[0] = len(targets)/np.sum(targets == 0)
        W[1] = len(targets)/np.sum(targets == 1)
        W = torch.Tensor(W).to(device=device)
        W = torch.log(torch.log(W) + 1)+1
        breakpoint()
        criterion = nn.CrossEntropyLoss(reduction='mean',weight=W)
        history = train(data_loader,model,optimizer,criterion,device,args.eval or args.test_path,args.epochs)



    