from torchvision import datasets, transforms
from pathlib import Path 
import torchvision

data_path = Path("data")

# TODO: Hardcode the vit transforms so that we don't have to download the weights to get the data
vit_weights = torchvision.models.ViT_B_32_Weights.DEFAULT
vit_transforms = vit_weights.transforms()

vit_train_transforms = transforms.Compose([
    transforms.TrivialAugmentWide(),
    vit_transforms
])

vit_train_data = datasets.FGVCAircraft(root=data_path,
                                       split="train",
                                       transform=vit_train_transforms,
                                       download=True)

vit_test_data = datasets.FGVCAircraft(root=data_path,
                                      split="test",
                                      transform=vit_transforms,
                                      download=True)
