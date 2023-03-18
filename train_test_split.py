import argparse
import os
import numpy as np
import shutil
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Train test split')
parser.add_argument('--data', type=str, default='training_data/30', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--ratio', type= float, default=0.8, help="to ratio of train w.r.t test")
parser.add_argument('--output', type=str, default='evaluation')


args = parser.parse_args()
class0 = os.listdir(args.data+'/0')
class1 = os.listdir(args.data+'/1')
class0 = ['0/'+x for x in class0]
class1 = ['1/'+x for x in class1]
allclass = class0 + class1

allclass = list(np.random.permutation(allclass))
train = allclass[:int(len(allclass)*args.ratio)]
test = allclass[-(len(allclass) -int(len(allclass)*args.ratio)):]
train_vf = ["/".join(args.output.split("/")+["train"] +x.split("/")[-2:]) for x in  train]
test_vf = ["/".join(args.output.split("/") +["test"]+x.split("/")[-2:]) for x in  test]
allclass = [args.data+'/'+x for x in allclass]
allclass_vf = train_vf + test_vf
for (a,b) in tqdm(zip(allclass,allclass_vf)):
    os.makedirs(os.path.dirname(b), exist_ok=True)
    shutil.copy(a,b)