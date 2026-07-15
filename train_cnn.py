import torchvision
import torch
from torchvision import datasets, transforms
from torchvision.transforms import v2
import torch.nn as nn
import wandb
from tqdm import tqdm
from pathlib import Path 
import time
import gc
from torch.utils.data import DataLoader


lr = 1e-2 
lr_decay = 0.9
EPOCHS = 75
vit_data_dir=Path("./data")
NUM_WORKERS = 0 
BATCH_SIZE = 32

vit_weights = torchvision.models.ViT_B_32_Weights.DEFAULT

transforms = v2.Compose([
    v2.CenterCrop(224),
    v2.Resize(256),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.ToTensor()
])


train_data = datasets.ImageFolder(root="../data/train", transform=transforms, target_transform=None)
test_data = datasets.ImageFolder(root="../data/test", transform=transforms, target_transform=None)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

if torch.backends.mps.is_available():
    device = "mps"
    print("[INFO] Using MPS (Metal Performance Shaders)")
elif torch.cuda.is_available():
    device = "cuda"
    print("[INFO] Using CUDA")
else:
    device = "cpu"
    print("[INFO] Using CPU")

def create_cnn(device,
               lr, 
               lr_decay,
               num_classes: int = 100,
               seed: int = 42):
    
    torch.manual_seed(seed)
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    

    return model, parameters

model = create_cnn(device=device, lr=lr, lr_decay=lr_decay)
print(f"[INFO] Model created and moved to {device}")
print(f"[INFO] Number of parameter groups: {len(parameters)}")
for i, param_group in enumerate(parameters):
    print(f"[INFO] Parameter group {i}: lr={param_group['lr']}, params={len(param_group['params'])}")

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader:  torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device,
          epochs: int = 10,
          save_model: bool = False,
          save_model_path: str = "./models",
          model_name: str = "vit_model"):
    
    if device == "mps":
        torch.mps.empty_cache()
        gc.collect()
        print("[INFO] MPS cache cleared and garbage collected")
    
    print("[INFO] Initializing wandb...")
    wandb.init(project="AircraftDetection", config={"epochs": epochs, "model name":model_name})

    log_interval = 10

    for epoch in tqdm(range(epochs)):
        model.to(device)
        model.train()

        train_loss, train_acc = 0, 0
        running_loss, running_acc = 0, 0
        start_time = time.time()

        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            acc = (y_pred_class == y).sum().item()/len(y_pred)
            train_acc += acc
            running_acc += acc

            if device == "mps" and batch % 5 == 0:
                torch.mps.empty_cache()
                if batch % 20 == 0:
                    gc.collect()

                if (batch + 1) % log_interval == 0 or (batch + 1) == len(train_dataloader):
                    elapsed = time.time() - start_time
                    avg_loss = running_loss / log_interval if (batch + 1) % log_interval == 0 else running_loss / ((batch + 1) % log_interval)
                    avg_acc = running_acc / log_interval if (batch + 1) % log_interval == 0 else running_acc / ((batch + 1) % log_interval)
                    print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch+1}/{len(train_dataloader)}] "
                          f"Avg Loss: {avg_loss:.4f} Avg Acc: {avg_acc:.4f} "
                          f"Elapsed: {elapsed:.1f}s")
                    running_loss, running_acc = 0, 0
                    start_time = time.time()
                    

        train_loss = train_loss / len(train_dataloader)
        train_acc = train_acc / len(train_dataloader)
        print(f"[INFO] Epoch {epoch+1} training complete. Avg train_loss: {train_loss:.4f}, Avg train_acc: {train_acc:.4f}")

        model.eval()
        test_loss, test_acc = 0, 0
        with torch.inference_mode():
            for batch, (X, y) in enumerate(test_dataloader):
                X, y = X.to(device), y.to(device)
                if batch == 0:
                    print("[INFO] First test batch loaded and moved to device.")

                # Forward pass
                test_pred_logits = model(X)
                if batch == 0:
                    print("[INFO] Forward pass complete for first test batch.")

                # Calculate and accumulate loss
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()
                if batch == 0:
                    print(f"[INFO] Test loss computed for first batch: {loss.item():.4f}")

                # Calculate and accumulate accuracy
                test_pred_labels = test_pred_logits.argmax(dim=1) 
                acc = ((test_pred_labels == y).sum().item() / len(test_pred_labels))
                test_acc += acc
                if batch == 0:
                    print(f"[INFO] Test accuracy computed for first batch: {acc:.4f}")
                    print(f"[INFO] First test batch - Predicted classes: {test_pred_labels[:5]}")
                    print(f"[INFO] First test batch - True labels: {y[:5]}")
                    print(f"[INFO] First test batch - Prediction probabilities (max): {torch.softmax(test_pred_logits, dim=1).max(dim=1)[0][:5]}")

            test_loss = test_loss / len(test_dataloader)
            test_acc = test_acc / len(test_dataloader)
            print(f"[INFO] Evaluation complete. Avg test_loss: {test_loss:.4f}, Avg test_acc: {test_acc:.4f}")
            
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

            print("[INFO] Logging metrics to wandb.")
            wandb.log({"test_loss": test_loss, "test_acc": test_acc, "train_loss": train_loss, "train_acc": train_acc})

    if save_model == True:
        print(f"[INFO] Saving {model_name} model to {save_model_path}")
        MODEL_PATH = Path(save_model_path) 
        MODEL_PATH.mkdir(parents=True,
                         exist_ok=True)
        MODEL_SAVE_PATH = MODEL_PATH/model_name
        torch.save(obj=model.state_dict(),
                   f=MODEL_SAVE_PATH)
        print(f"[INFO] Model saved to {MODEL_SAVE_PATH}")
        return results
    else:
        print("[INFO] Training complete. Model not saved.")
        return results


vit_loss_fn = torch.nn.CrossEntropyLoss()
vit_optimizer = torch.optim.Adam(params=parameters)

train(model=model,
      train_dataloader=train_dataloader,
      test_dataloader=train_dataloader,
      loss_fn=vit_loss_fn,
      optimizer=vit_optimizer,
      device=device,
      epochs=EPOCHS,
      save_model=True,
      save_model_path="./models",
      model_name=f"layer_wise_learning_rate")
