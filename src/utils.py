import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 			WHAT DOES THIS DO !?
# 			---------------------
# 			This script has two visualization utility functions
#
# 			plot_training_curves - Creates a side by side plot of
# 				training losses and accuracies. It shows the model's
# 				learning progress over epochs and saves the plot as
# 				'training_curves.png'
#
# 			plot_confusion_matrix - This creates a heatmap showing the
# 				models classification performance. The heatmap shows the
# 				counts of true vs predicted labels and helps identify where
# 				mistakes are made. Saves as 'confusion_matrix.png'
#
# 			Thanks matplotlib and seaborn for doing the heavy lifting on visualization
# 			and saving the results where they need to go.

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Losses')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot accuracies
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracies')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('results/training_curves.png')
    plt.close()

def plot_confusion_matrix(true_labels, predictions, classes):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.close()
