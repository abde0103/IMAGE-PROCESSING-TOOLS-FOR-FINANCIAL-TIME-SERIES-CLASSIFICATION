
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from tqdm import tqdm
from cnn import Cnn

parser = argparse.ArgumentParser(description='training and cross validation')
parser.add_argument('--optim', type = str, help='select between sgd or adam', default = 'sgd' )
parser.add_argument('--augment_data', action = 'store_true' ,help = 'if you want to apply data augmentation')
parser.add_argument('--tensorboard_log_dir', type = str, help = 'path for tensorboard output', required=True)

parser.add_argument('--data', type=str, default='example', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
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
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Data initialization and loading
from data import data_transforms
from data import data_augmentation

if args.augment_data:
  dataset1 = datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms)
  dataset2 = datasets.ImageFolder(args.data + '/train_images',
                         transform=data_augmentation)
  f_dataset = torch.utils.data.ConcatDataset([dataset1,dataset2])
  classes = set(dataset1.targets)
  n_classes = len(classes)

else:
  f_dataset =  datasets.ImageFolder(args.data + '/train_images',
                        transform=data_transforms)
  classes = set(f_dataset.targets)
  n_classes = len(classes)                  

if args.final:
  print('final training')
  val_dataset =  datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms)
  train_loader = torch.utils.data.DataLoader( torch.utils.data.ConcatDataset([f_dataset,val_dataset])
      ,
      batch_size=args.batch_size, shuffle=True, num_workers=1)
else:
  train_loader = torch.utils.data.DataLoader( f_dataset
      ,
      batch_size=args.batch_size, shuffle=True, num_workers=1)
  val_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(args.data + '/val_images',
                          transform=data_transforms),
      batch_size=args.batch_size, shuffle=False, num_workers=1)





model = Cnn(n_classes, args.only_fc_layer, args.auto_path)

if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

if args.optim == 'sgd':
  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optim == 'adam':
  optimizer = optim.Adam(model.parameters(),lr = args.lr, weight_decay= args.weight_decay)

from torch.utils.tensorboard import SummaryWriter
exp = 'res152_LR_{}_mom_{}_only_{}_batch_size_{}_augmented_{}_optim_{}_weight_{}'.format(args.lr,args.momentum,args.only_fc_layer, \
args.batch_size,args.augment_data, args.optim, args.weight_decay)
tf_dir = args.tensorboard_log_dir+exp
writer = SummaryWriter(log_dir = tf_dir)
step = 0
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma = 0.5, verbose = True)
def train(epoch,step):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
        if step % args.log_interval == 0:
          writer.add_scalar('train_loss',loss.data.item(),step)
        step +=1

    return step


def validation(epoch):
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    validation_loss, correct, len(val_loader.dataset),
    100. * correct / len(val_loader.dataset)))
    writer.add_scalar('validation_loss',validation_loss, epoch)
    writer.add_scalar('validation accuracy',correct/ len(val_loader.dataset),epoch)


for epoch in range(1, args.epochs + 1):
    step = train(epoch, step)
    scheduler.step()
    if not args.final:
      validation(epoch)
      model_file = args.experiment + '/model_'+exp+'_epoch_' + str(epoch) + '.pth'
      torch.save(model.state_dict(), model_file)
      print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
if args.final:
  model_file = args.experiment + '/final_model_'+exp+'_epoch_' + str(epoch) + '.pth'
  torch.save(model.state_dict(), model_file)
  print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')


