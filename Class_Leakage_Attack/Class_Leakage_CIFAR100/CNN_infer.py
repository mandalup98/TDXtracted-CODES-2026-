import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
import os
import time

# ✅ Load CIFAR-100 Class Names
dataset_tmp = CIFAR100(root='./data', train=False, download=True)
CIFAR_CLASSES = {i: name for i, name in enumerate(dataset_tmp.classes)}

class CIFAR100_CNN(nn.Module):
    def __init__(self):
        super(CIFAR100_CNN, self).__init__()
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
            nn.Linear(512, 100)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

parser = argparse.ArgumentParser(description="Run inference on a specific CIFAR-100 image 100 times")
parser.add_argument("--class_index", type=int, required=True, help="Class index (0-99) for CIFAR-100")
parser.add_argument("--image_index", type=int, required=True, help="Index of the image within the class")
args = parser.parse_args()

if args.class_index not in CIFAR_CLASSES:
    raise ValueError("Invalid class index. Must be between 0–99.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFAR100_CNN().to(device)
model.load_state_dict(torch.load("cifar100_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
])

test_dataset = CIFAR100(root='./data', train=False, transform=transform, download=True)

# ✅ Filter test set for selected class
indices = [i for i, (_, label) in enumerate(test_dataset) if label == args.class_index]

if args.image_index >= len(indices):
    raise ValueError(f"Invalid image index. Max available for class {args.class_index} is {len(indices)-1}.")

selected_image_index = indices[args.image_index]
image, label = test_dataset[selected_image_index]

image = image.unsqueeze(0).to(device)

print(f"\n🔍 Inference on CIFAR-100 class '{CIFAR_CLASSES[args.class_index]}' (Index: {args.class_index}) - Image {args.image_index}:\n")

start_time = time.time()
run = 0
while time.time() - start_time < 200:
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1).item()
        #print(f"Run {i+1}: Predicted Label = {CIFAR_CLASSES[pred]}, True Label = {CIFAR_CLASSES[label]}")

print("\n✅ Inference Complete!")
os.system("shutdown -h now")
