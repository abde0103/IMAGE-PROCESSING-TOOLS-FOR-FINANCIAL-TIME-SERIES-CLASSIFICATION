
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from torchvision import datasets
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, SequentialSampler, WeightedRandomSampler
import matplotlib.pyplot as plt
from typing import Any, Callable, Optional
from PIL import Image

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
   
    return pil_loader(path)

def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)

class MyData(datasets.ImageFolder):
    def __init__(self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        transform2 = None):

        super().__init__(root,transform,target_transform,loader,is_valid_file)
        self.transform2 = transform2


    def __getitem__(self, index):
        x,y = super().__getitem__(index)
        if y == 0:
            x = self.transform2(x)
        return x, y

def get_transform(resize):
    data_transform = transforms.Compose(
    [transforms.Resize((resize, resize)),
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    )
    return data_transform

def get_transform_augmentation():
    data_transforms = transforms.Compose(
        [transforms.RandomErasing(0.3)]
    )
    return data_transforms

def get_train_data_loader(data_path,batch_size,shuffle= True,resize= 50):
    print("getting eval data loader...")
    data_transform = get_transform(resize)
    data_augmentation= get_transform_augmentation()
    dataset = MyData(data_path,transform=data_transform,transform2=data_augmentation)
    w = np.zeros(2)
    temp = np.array(dataset.targets)
    w[0] =1/np.sum(temp == 0)
    w[1] = 1/np.sum(temp == 1)
    W = np.zeros(len(temp))
    W[np.where(temp == 0)[0]] = 1 #w[0]
    W[np.where(temp == 1)[0]] = 1 #w[1]
    sampler = WeightedRandomSampler(W,int(2*len(W)))
    class_names = dataset.classes
    print(f"Train image size : {len(dataset)}")
    print(f"train classes: {class_names}")
    dataloader = {'train':[DataLoader(dataset, batch_size,sampler=sampler)],'val':[]}
    return dataloader,class_names


def  get_eval_data_loader(data_path, batch_size, shuffle= True,cv = 0,resize = 50):
    print("getting eval data loader...")
    data_transform = get_transform(resize)
    data_augmentation= get_transform_augmentation()
    dataset =  MyData(data_path,transform=data_transform,transform2= data_augmentation)
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
        w = np.zeros(2)
        temp = np.array(dataset.targets)
        w[0] =1/np.sum(temp == 0)
        w[1] = 1/np.sum(temp == 1)
        W = np.zeros(len(temp))
        W[np.where(temp == 0)[0]] = 1 #w[0]
        W[np.where(temp == 1)[0]] = 1 #w[1]
        W[val_idx] = 0
        #train_sampler = SequentialSampler(train_idx)
        train_sampler = WeightedRandomSampler(W,int(2*len(train_idx)))
        #train_sampler = SubsetRandomSampler(train_idx)
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


        