
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from torchvision import datasets
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
import matplotlib.pyplot as plt


def get_transform(resize):
    data_transform = transforms.Compose(
    [transforms.Resize((resize, resize)),
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    )
    return data_transform

def get_train_data_loader(data_path,batch_size,shuffle= True,resize= 50):
    print("getting eval data loader...")
    data_transform = get_transform(resize)
    dataset = datasets.ImageFolder(data_path,transform=data_transform)
    class_names = dataset.classes
    print(f"Train image size : {len(dataset)}")
    print(f"train classes: {class_names}")
    dataloader = {'train':[DataLoader(dataset, batch_size,shuffle)],'val':[]}
    return dataloader,class_names


def  get_eval_data_loader(data_path, batch_size, shuffle= True,cv = 0,resize = 50):
    print("getting eval data loader...")
    data_transform = get_transform(resize)
    dataset = datasets.ImageFolder(data_path,transform=data_transform)
    class_names = dataset.classes
    print(f"train classes: {class_names}")
    if cv <= 1:
        splits=KFold(n_splits=4,shuffle=True,random_state=42)
    else:
        splits=KFold(n_splits=cv,shuffle=True,random_state=42)
    data_loader = {'train':[], 'val':[]}
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        if fold == 0:
            print(f"Train image size : {len(train_idx)}")
            print(f"val image size : {len(val_idx)}")
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        data_loader['train'].append( DataLoader(dataset, batch_size=batch_size, sampler=train_sampler))
        data_loader['val'].append(DataLoader(dataset, batch_size=batch_size, sampler=val_sampler))
        if cv <= 1:
            break
    return data_loader,class_names

def get_test_data_loader(data_path, batch_size, shuffle = False,resize = 50):
    data_transform = get_transform(resize)
    dataset = datasets.ImageFolder(data_path,transform=data_transform)
    data_loader = DataLoader(dataset,batch_size,shuffle)
    class_names = dataset.classes
    print(f"test classes: {class_names}")
    print(f"test image size : {len(dataset)}")
    return data_loader,class_names

def plot_imageTensor(tensor):
    img = transforms.functional.to_pil_image(tensor)
    plt.imshow(img)
    plt.show(block = False ) 


        