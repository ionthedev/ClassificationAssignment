import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# 			WHAT DOES THIS DO !?
# 			---------------------
# 			This script uses TorchVision to manage the CIFAR10 datasets
# 			and applies consistent transformations to all of the images,
# 			ensuring all of the images are able to be processed in the
# 			same way. The batched data is then efficiently loaded for model training.

class CIFAR10DataLoader:
    def __init__(self, batch_size=32, val_split=0.1):
        self.batch_size = batch_size # Sets the batch size to a default of 32
        self.val_split = val_split # Set the validation ratio to 10%

        # This then converts the tensor and normalizes it with a mean or std of 0.5
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def get_data_loaders(self):
        # Downloads the dataset and splits it into train and validation sets for us, creating a full dataset as well as a test dataset
        full_dataset = datasets.CIFAR10(root='./data',
                                      train=True,
                                      download=True,
                                      transform=self.transform)

        test_dataset = datasets.CIFAR10(root='./data',
                                      train=False,
                                      download=True,
                                      transform=self.transform)

        # This sets the size of the training and validation sets so we have an actual limit
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size

        # ACTUALLY splits the dataset based on the size we just defined
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

        # Create the data loaders for the training, validation and the test datasets
        train_loader = DataLoader(train_dataset,
                                batch_size=self.batch_size,
                                shuffle=True) # this shuffles the set like a deck of cards

        val_loader = DataLoader(val_dataset,
                              batch_size=self.batch_size,
                              shuffle=False) # this and ---------|
        													 #   |
        test_loader = DataLoader(test_dataset,				 #   |-  doesn't shuffle the set.
                               batch_size=self.batch_size,   #   |
                               shuffle=False) # this ------------|

        return train_loader, val_loader, test_loader
