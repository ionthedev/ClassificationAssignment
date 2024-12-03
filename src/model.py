import torch.nn as nn

# 			WHAT DOES THIS DO !?
# 			---------------------
# 			This script sets the definition for the CNN (Convolutional Neural Network).
# 			CNN's look at things in layers. The __init__ function creates these layers
# 			with increasing filters (3->32->64->64). We need to set the max pooling to
# 			respect to reduce dimensions after each conv layer. We then use rectified
# 			linear unit activation between the layers.
#
#			So basically the image passes through the layers (conv+ReLU+pool).
#			Flattens to 1 Dimension for view operations.
#			Passes that through fully connected layers and then outputs 10 class
#			probabilities, the CIFAR-10 classes. The final output shape will match the 10
#			classes defined in CIFAR-10.

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 32x16x16
        x = self.pool(self.relu(self.conv2(x)))  # 64x8x8
        x = self.pool(self.relu(self.conv3(x)))  # 64x4x4
        x = x.view(-1, 64 * 4 * 4)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
