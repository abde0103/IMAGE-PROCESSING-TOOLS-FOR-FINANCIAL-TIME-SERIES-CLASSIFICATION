
import torchvision.transforms as transforms


data_augmentation = transforms.Compose(
  [transforms.Resize((232,232)),
  transforms.RandomCrop(224),
  transforms.RandomHorizontalFlip(0.5),
  transforms.RandomRotation((0,90))
  ]
)