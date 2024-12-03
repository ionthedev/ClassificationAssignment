import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 			WHAT DOES THIS DO !?
# 			---------------------
# 			This script sets up the training basics. Initializing the loss
#			function (CrossEntropyLoss) and Adam Optimizer. It then cycles
#			through the number of epochs(10 by default), performing both
#			training and validitation phases. While it's training, it processes
#			batches of data, calculates losses, performs backpropogation and
#			and updates weights accordingly before validating.
#			Validation then evaluates the performance without updating the weights
#			Through the whole training process, it is keeping track and stores loss
#			and accuracy metrics for both the phases it goes through. Then tqdm
#			displays progress with progress bars and prints out the performance
#			metrics after each epoch, returing the collected metrics for visualization.


def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    model = model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%')

    return train_losses, val_losses, train_accs, val_accs
