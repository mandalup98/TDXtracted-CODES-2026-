import torch, intel_extension_for_pytorch as ipex
from transformers import AutoImageProcessor, AutoModelForImageClassification   # newer API
# (your original imports are fine; AutoImageProcessor is just the modern name)
from PIL import Image
import requests
import os
os.environ["ONEDNN_VERBOSE"] = "0"

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/swin-base-patch4-window7-224"
).eval()

# Optional but helpful for conv/patch-embed paths
model = model.to(memory_format=torch.channels_last)

# Enable BF16 execution on CPU; oneDNN will pick AMX BF16 kernels on 4th-gen Xeon
model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)

# Prepare inputs
inputs = processor(images=image, return_tensors="pt")
# Put pixel tensor in channels_last layout (good for conv-heavy parts)
pixel_values = inputs["pixel_values"].to(memory_format=torch.channels_last)

with torch.no_grad(), torch.cpu.amp.autocast(dtype=torch.bfloat16):
    for i in range(0,100000):
        outputs = model(pixel_values=pixel_values)

logits = outputs.logits
pred = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[pred])

