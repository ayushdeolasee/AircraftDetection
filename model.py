import torchvision
import torch
import torch.nn as nn

def create_vit(device,
               num_classes: int = 100,
               seed: int = 42):
    weights = torchvision.models.ViT_B_32_Weights.DEFAULT
    model = torchvision.models.vit_b_32(weights=weights).to(device)

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    
    model.heads = nn.Sequential(
        nn.Linear(in_features=768, out_features=num_classes)
    )

    return model
