import argparse


parser = argparse.ArgumentParser(description='training and cross validation')
parser.add_argument('--tensorboard_log_dir', type = str, help = 'path for tensorboard output', required=True)
parser.add_argument('--optim', type = str, help='select between sgd or adam', default = 'sgd' )

parser.add_argument('--data', type=str, default='example/360', metavar='D',
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
parser.add_argument('--eval', type=bool, action='store_true' , metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--test_path', type=str,  metavar='E',
                    help='folder where experiment outputs are located.')

def cross_val(**kwargs):
    params_names = []
    values = []
    for params_name,value in kwargs.items():
        params_names.append(params_name)
        values.append(value)
    
    values = n

