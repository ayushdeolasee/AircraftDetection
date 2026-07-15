import os
from torchvision.transforms import v2
import torchvision.io as io
import random
from torchvision import datasets
import torchvision
from pathlib import Path

data_path = Path("data")

vit_weights = torchvision.models.ViT_B_32_Weights.DEFAULT

vit_transforms = v2.Compose([
    v2.CenterCrop(224),
    v2.Resize(256),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

vit_train_transforms = v2.Compose([
    v2.TrivialAugmentWide(),
    vit_transforms, 
])

print("Downlaoding Train Dataset...")
vit_train_data = datasets.FGVCAircraft(root=data_path,
                                       split="train",
                                       transform=vit_train_transforms,
                                       download=True)

print("Downlaoding Test Dataset...")
vit_test_data = datasets.FGVCAircraft(root=data_path,
                                      split="test",
                                      transform=vit_transforms,
                                      download=True)

print("Downlaoding Val Dataset...")
vit_val_data = datasets.FGVCAircraft(root=data_path,
                                      split="val",
                                      transform=vit_transforms,
                                      download=True)

train_images = {}
test_images = {}
variants = []

print("Reading Variants...")
with open("./data/fgvc-aircraft-2013b/data/variants.txt", "r") as f:
    data = f.readlines()
    for line in data:
        variants.append(line.replace("\n", "").replace(" ", "_").replace("/", "_"))

with open("./data/fgvc-aircraft-2013b/data/images_variant_trainval.txt", "r") as f:
    data = f.readlines()
    for line in data:
        image_data = line.split(" ", 1)
        train_images[image_data[0]] = image_data[1].replace(" ", "_").replace("/", "_").replace("\n", "")

with open("./data/fgvc-aircraft-2013b/data/images_variant_test.txt", "r") as f:
    data = f.readlines()
    for line in data:
        image_data = line.split(" ", 1)
        test_images[image_data[0]] = image_data[1].replace(" ", "_").replace("/", "_").replace("\n", "")


print("Creating Directories...")
if not os.path.exists("./data/test"):
   os.mkdir("./data/test") 

if not os.path.exists("./data/train"):
   os.mkdir("./data/train") 

for variant in variants:
    os.mkdir(f"./data/train/{variant}")
    os.mkdir(f"./data/test/{variant}")

print("Moving Train Images...")
for image in train_images:
    old_image_path = f"./data/fgvc-aircraft-2013b/data/images/{image}.jpg"
    varaint = train_images[image]
    new_image_path = f"./data/train/{varaint}/{image}.jpg"
    os.rename(old_image_path, new_image_path)

print("Moving Test Images...")
for image in test_images:
    old_image_path = f"./data/fgvc-aircraft-2013b/data/images/{image}.jpg"
    varaint = test_images[image]
    new_image_path = f"./data/test/{varaint}/{image}.jpg"
    os.rename(old_image_path, new_image_path)



transforms = [
    # Transform 1: Color & Geometric Augmentation (Enhanced)
    v2.Compose([
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.85, 1.15), shear=8),
        v2.RandomPerspective(distortion_scale=0.3, p=0.4),
        v2.RandomErasing(p=0.3, scale=(0.02, 0.25), ratio=(0.3, 3.3), value=0),
        v2.ToPILImage()
    ]), 
    
    # Transform 2: Blur & Geometric Focus
    v2.Compose([
        v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.5)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.3),
        v2.RandomRotation(degrees=12),
        v2.RandomAffine(degrees=8, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        v2.RandomPerspective(distortion_scale=0.4, p=0.5),
        v2.ToPILImage()
    ]),
    
    # Transform 3: High Contrast & Sharp Details
    v2.Compose([
        v2.ColorJitter(brightness=0.2, contrast=0.6, saturation=0.5, hue=0.05),
        v2.RandomAdjustSharpness(sharpness_factor=2.0, p=0.7),
        v2.RandomAutocontrast(p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=8),
        v2.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=3),
        v2.RandomErasing(p=0.2, scale=(0.01, 0.15), ratio=(0.5, 2.0), value='random'),
        v2.ToPILImage()
    ]),
    
    # Transform 4: Atmospheric & Lighting Effects
    v2.Compose([
        v2.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.03),
        v2.RandomPosterize(bits=6, p=0.4),
        v2.RandomSolarize(threshold=128, p=0.3),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.2),
        v2.RandomRotation(degrees=20),
        v2.RandomAffine(degrees=12, translate=(0.12, 0.12), scale=(0.88, 1.12), shear=6),
        v2.RandomPerspective(distortion_scale=0.6, p=0.6),
        v2.RandomErasing(p=0.4, scale=(0.03, 0.2), ratio=(0.3, 3.0), value=128),
        v2.ToPILImage()
    ]),
    
    # Transform 5: Minimal & Clean Augmentation
    v2.Compose([
        v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        v2.RandomEqualize(p=0.3),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=5),
        v2.RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.97, 1.03), shear=2),
        v2.RandomErasing(p=0.1, scale=(0.01, 0.08), ratio=(0.5, 2.0), value=0),
        v2.ToPILImage()
    ])
]

print("Applying Transforms to Train Images...")
for variant in variants:
    with os.scandir(f"./data/train/{variant}") as images:
        for image in images:
            name = str(image.name).replace(".jpg", "")
            image_tensor = io.read_image(image.path)
            for transform in transforms:
                image = transform(image_tensor)
                image.save(f"./data/train/{variant}/{name}{str(random.randint(1, 1000000))}.jpg")
