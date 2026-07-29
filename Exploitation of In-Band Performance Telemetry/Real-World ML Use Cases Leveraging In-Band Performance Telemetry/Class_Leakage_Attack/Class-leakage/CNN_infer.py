import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
import numpy as np
import time
import os

# ✅ Define CIFAR-10 Class Mapping
CIFAR_CLASSES = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 
    5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}

# ✅ Load Model
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# ✅ Command Line Argument Parsing
parser = argparse.ArgumentParser(description="Run inference on a specific CIFAR-10 image 100 times")
parser.add_argument("--class_index", type=int, required=True, help="Class index (0-9) for CIFAR-10")
parser.add_argument("--image_index", type=int, required=True, help="Index of the image within the class")
args = parser.parse_args()

# ✅ Check Validity of Class Index
if args.class_index not in CIFAR_CLASSES:
    raise ValueError("Invalid class index. Must be between 0-9.")

# ✅ Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFAR10_CNN().to(device)
model.load_state_dict(torch.load("cifar10_cnn.pth"))
model.eval()

# ✅ Load CIFAR-10 test set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

test_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)

# ✅ Filter test set for selected class
class_label = args.class_index
indices = [i for i, (_, label) in enumerate(test_dataset) if label == class_label]

# ✅ Ensure image index is within range
if args.image_index >= len(indices):
    raise ValueError(f"Invalid image index. Max available for class {class_label} is {len(indices)-1}.")

selected_image_index = indices[args.image_index]
image, label = test_dataset[selected_image_index]

# ✅ Prepare image for inference
image = image.unsqueeze(0).to(device)  # Add batch dimension

# ✅ Run Inference 100 Times

print(f"\n🔥 Running inference on class '{CIFAR_CLASSES[class_label]}' (Index: {args.class_index}) - Image {args.image_index} (True Label: {label}) 100 times...\n")

start_time = time.time()

while time.time() - start_time < 165:
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).item()
        #print(f"Run {i+1}: Predicted Label = {pred}, True Label = {label}")

print("\n✅ Inference Complete!")
#os.system("shutdown -h now")
