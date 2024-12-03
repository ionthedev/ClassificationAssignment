import torch
from src.data_loader import CIFAR10DataLoader
from src.model import SimpleCNN
from src.train import train_model
from src.utils import plot_training_curves, plot_confusion_matrix

# 			WHAT DOES THIS DO !?
# 			---------------------
# 			what DOESN't it do is a better question. All jokes aside,
# 			This script sets up the devices (GPU/CPU) and initializes the
# 			data loaders with a batch size of 64. It then creates the CNN
# 			instance and trains it for 10 epochs, collecting metrics along
# 			the way. It then plots the training and validation curves and
# 			evaluates the model on the test set. When it's all done, it
# 			creates a confusion matrix to show how dumb it is in regards
# 			to all 10 CIFAR classes.

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    data_loader = CIFAR10DataLoader(batch_size=64)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()

    # Initialize model
    model = SimpleCNN()

    # Train model
    metrics = train_model(model, train_loader, val_loader, num_epochs=10, device=device)
    train_losses, val_losses, train_accs, val_accs = metrics

    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    # Evaluate on test set
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Plot confusion matrix
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']
    plot_confusion_matrix(all_labels, all_preds, classes)

if __name__ == '__main__':
    main()
