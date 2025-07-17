import torch
import torchvision
from torchvision import datasets
from pathlib import Path 
from torch.utils.data import DataLoader
import os
from tqdm.auto import tqdm
import wandb
from torch import nn

NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 64
lr = 0.2
EPOCHS = 150

device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"

vit_weights = torchvision.models.ViT_B_32_Weights.DEFAULT
vit_transforms = vit_weights.transforms()
vit_data_dir = Path("vit_data")

vit_train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.TrivialAugmentWide(),
    vit_transforms 
])

# Getting ViT traning data
vit_train_data = datasets.FGVCAircraft(root=vit_data_dir,
                                       split="train",
                                       transform=vit_train_transforms,
                                       download=True)

# Get ViT test data
vit_test_data = datasets.FGVCAircraft(root=vit_data_dir,
                                      split="test",
                                      transform=vit_transforms,
                                      download=True)


vit_train_dataloder = DataLoader(dataset=vit_train_data,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=NUM_WORKERS,
                                 pin_memory=True)

vit_test_dataloder = DataLoader(dataset=vit_test_data,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=NUM_WORKERS,
                                 pin_memory=True)

num_classes = len(vit_train_data.classes)

from torch import nn

def create_vit(device,
               num_classes: int = 100,
               seed: int = 42):
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    model = torchvision.models.vit_b_16(weights=weights).to(device)

    # Freeze feature extraction layers
    # for param in model.parameters():
        # param.requires_grad = False

    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    # Create a new classifier layer
    
    model.heads = nn.Sequential(
        nn.Linear(in_features=768, out_features=num_classes)
    )

    return model

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader:  torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device,
          learning_rate,
          epochs: int = 10,
          save_model: bool = False,
          save_model_path: str = "./models",
          model_name: str = "vit_model"):
    
    wandb.init(project="AircraftDetection", config={"epochs": epochs, "model name":model_name, "learning rate": learning_rate})

    for epoch in tqdm(range(epochs)):
        # Put the model in train mode
        model.to(device)
        model.train()

        # Setup train loss and train accuracy values
        train_loss, train_acc = 0, 0

        for batch, (X, y) in enumerate(train_dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_pred = model(X)

            # Calculate and accumulate loss 
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            # Optimizer zero grad
            optimizer.zero_grad()

            # Loss backward
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Calculate and accumulate accuray metrics across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item()/len(y_pred)

        # Adjust metrics to get average loss and accuracy per batch
        train_loss = train_loss / len(train_dataloader)
        train_acc = train_acc / len(train_dataloader)

        # Put the model in eval mode
        model.to(device)
        model.eval()

        # Setup test loss and test accuracy values
        test_loss, test_acc = 0, 0

        with torch.inference_mode():
            # Loop through DataLoader batches
            for batch, (X, y) in enumerate(test_dataloader):
                # Send data to target deivce
                X, y = X.to(device), y.to(device)

                # Forward pass
                test_pred_logits = model(X)

                # Calculate and accumulate loss
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()

                # Calculate and accumulate accuracy
                test_pred_labels = test_pred_logits.argmax(dim=1) 
                test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

            # Adjust metrics to get average loss and accuracy per batch
            test_loss = test_loss / len(train_dataloader)
            test_acc = test_acc / len(test_dataloader)
            
            results = {"train_loss": [],
                        "train_acc": [],
                        "test_loss": [],
                        "test_acc": []}
            print(
                    f"Epoch: {epoch+1} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"train_acc: {train_acc:.4f} | "
                    f"test_loss: {test_loss:.4f} | "
                    f"test_acc: {test_acc:.4f}")
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            wandb.log({"test_loss": test_loss, "test_acc": test_acc, "train_loss": train_loss, "train_acc": train_acc})

    if save_model == True:
        print(f"[INFO] Saving {model_name} model to {save_model_path}")
        MODEL_PATH = Path(save_model_path) 
        MODEL_PATH.mkdir(parents=True,
                         exist_ok=True)
        MODEL_SAVE_PATH = MODEL_PATH/model_name
        torch.save(obj=model.state_dict(),
                   f=MODEL_SAVE_PATH)
        return results
    else:
        return results

vit = create_vit(device=device, num_classes=num_classes, seed=42)
vit.to(device)
vit_loss_fn = torch.nn.CrossEntropyLoss()
vit_optimizer = torch.optim.Adam(params=vit.parameters(), lr=lr)

train(model=vit,
      train_dataloader=vit_train_dataloder,
      test_dataloader=vit_test_dataloder,
      loss_fn=vit_loss_fn,
      learning_rate=lr,
      optimizer=vit_optimizer,
      device=device,
      epochs=EPOCHS,
      save_model=True,
      save_model_path="./models",
      model_name=f"vit_{EPOCHS}_epochs")