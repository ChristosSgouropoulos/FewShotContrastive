import os 
import sys 
import json
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from models.main_models import EncoderLinearSoftmax
from datasets.miniimagenet import MiniImageNet
from tqdm import tqdm



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
mini_imagenet_84_path = "../../../data/mini_imagenet/images"
experiment_config = "src/experiment_config.json"
with open(experiment_config, "r") as f:
        config_file = json.load(f) 
training_set_84 = MiniImageNet(root = mini_imagenet_84_path, split = "train")
training_set,validation_set = training_set_84.split_train_validation(train_percentage=0.8)

train_dataloader = DataLoader(training_set, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size= 64, shuffle = True)

model = EncoderLinearSoftmax(config_file = config_file)
model.to(device)

optimizer = torch.optim.SGD(params = model.parameters(), 
                            lr=0.0001,
                              momentum=0.9, 
                              weight_decay=0.0005, nesterov=True)
loss_fn = torch.nn.CrossEntropyLoss()


print("*****************************************")



## Training 

epochs = 150

# Training loop

save_dir = "saved_encoders"
os.makedirs(save_dir, exist_ok=True)

# Initialize variables to track the best validation accuracy
best_val_acc = 0.0
best_epoch = -1

# Training loop
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Training phase
    for inputs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the GPU if available
        
        optimizer.zero_grad()  # Clear the gradients

        outputs = model(inputs)  # Forward pass
        loss = loss_fn(outputs, labels)  # Compute the loss

        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the model parameters

        running_loss += loss.item() * inputs.size(0)  # Accumulate loss
        _, preds = torch.max(outputs, 1)  # Get the predictions
        correct_predictions += torch.sum(preds == labels)  # Count correct predictions
        total_samples += labels.size(0)  # Update the number of samples

    epoch_loss = running_loss / total_samples  # Calculate epoch loss
    epoch_acc = correct_predictions.double() / total_samples  # Calculate epoch accuracy

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_running_loss = 0.0
    val_correct_predictions = 0
    val_total_samples = 0

    with torch.no_grad():  # Disable gradient calculation for validation
        for val_inputs, val_labels in validation_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            val_outputs = model(val_inputs)
            val_loss = loss_fn(val_outputs, val_labels)

            val_running_loss += val_loss.item() * val_inputs.size(0)
            _, val_preds = torch.max(val_outputs, 1)
            val_correct_predictions += torch.sum(val_preds == val_labels)
            val_total_samples += val_labels.size(0)

    val_epoch_loss = val_running_loss / val_total_samples
    val_epoch_acc = val_correct_predictions.double() / val_total_samples

    # Print training and validation statistics
    print(f"Epoch [{epoch+1}/{epochs}] | "
          f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f} | "
          f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.4f}")

    # Check if this is the best validation accuracy so far
    if val_epoch_acc > best_val_acc:
        best_val_acc = val_epoch_acc
        best_epoch = epoch + 1  # epochs are 0-indexed in code
        encoder_path = os.path.join(save_dir, f"encoder_epoch_{best_epoch}_val_acc_{best_val_acc:.4f}.pth")
        torch.save(model.encoder.state_dict(), encoder_path)
        print(f"New best validation accuracy! Saving encoder weights to {encoder_path}")

# Final message after training
print("Training completed.")
print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
