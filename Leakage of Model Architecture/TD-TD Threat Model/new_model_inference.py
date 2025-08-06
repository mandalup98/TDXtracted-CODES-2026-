import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
from PIL import Image
import argparse
import time
import os

# Define all requested models
MODEL_MAP = {
    "alexnet": models.alexnet,
    "vgg11": models.vgg11,
    "vgg13": models.vgg13,
    "vgg16": models.vgg16,
    "vgg19": models.vgg19,
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "squeezenet1_0": models.squeezenet1_0,
    "squeezenet1_1": models.squeezenet1_1,
    "densenet121": models.densenet121,
    "densenet161": models.densenet161,
    "densenet169": models.densenet169,
    "densenet201": models.densenet201,
    "inception_v3": models.inception_v3,
    "googlenet": models.googlenet,
    "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0,
    "mobilenet_v2": models.mobilenet_v2,
}

def load_model(model_name):
    """Load a pre-trained model, handling torchvision version compatibility."""
    if model_name not in MODEL_MAP:
        raise ValueError(f"Invalid model name. Choose from: {list(MODEL_MAP.keys())}")
    
    model_fn = MODEL_MAP[model_name]
    version = tuple(map(int, torchvision.__version__.split('.')[:2]))  # e.g. (0, 12)

    if version >= (0, 13):  # For torchvision >= 0.13
        try:
            weights_enum = getattr(models, f"{model_name.upper()}_Weights")
            model = model_fn(weights=weights_enum.IMAGENET1K_V1)
        except AttributeError:
            model = model_fn(weights="IMAGENET1K_V1")
    else:
        model = model_fn(pretrained=True)

    model.eval()
    return model

def preprocess_image(model_name=""):
    """Generate a random RGB image and apply transformations. Special case for inception_v3."""
    size = (299, 299) if model_name == "inception_v3" else (224, 224)
    image = Image.effect_noise(size, 100).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def run_inference(model, image_tensor, duration_seconds=20):
    """Run inference repeatedly for a fixed duration (default: 2 minutes)."""
    iterations = 0
    start_time = time.time()
    
    with torch.no_grad():
        while time.time() - start_time < duration_seconds:
            _ = model(image_tensor)
            iterations += 1

    print(f"Completed {iterations} iterations in {duration_seconds} seconds.")

def main():
    parser = argparse.ArgumentParser(description="Run inference for 2 minutes on a random image using a chosen model.")
    parser.add_argument("model", choices=MODEL_MAP.keys(), help="Choose a model for inference.")
    args = parser.parse_args()

    model = load_model(args.model)
    image_tensor = preprocess_image(args.model)

    print(f"Running inference for 2 minutes using {args.model}...")
    run_inference(model, image_tensor, 300)
    os.system("shutdown -h now")

if __name__ == "__main__":
    main()

