from torchvision import datasets, transforms
from pathlib import Path 
import torchvision

data_path = Path("data")

# TODO: Hardcode the vit transforms so that we don't have to download the weights to get the data
vit_weights = torchvision.models.ViT_B_32_Weights.DEFAULT

vit_train_transforms = transforms.Compose([
    transforms.TrivialAugmentWide(),
    transforms.CenterCrop(224),
    transforms.Resize(256),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

vit_transforms = 


vit_train_data = datasets.FGVCAircraft(root=data_path,
                                       split="train",
                                       transform=vit_train_transforms,
                                       download=True)

vit_test_data = datasets.FGVCAircraft(root=data_path,
                                      split="test",
                                      transform=vit_transforms,
                                      download=True)

vit_val_data = datasets.FGVCAircraft(root=data_path,
                                      split="val",
                                      transform=vit_transforms,
                                      download=True)
