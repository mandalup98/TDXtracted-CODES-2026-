import torch, intel_extension_for_pytorch as ipex
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
import os
os.environ["ONEDNN_VERBOSE"] = "0"

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224').eval()

# ↓ Optimize for BF16 on CPU (oneDNN will choose AMX BF16 kernels)
model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)

inputs = feature_extractor(images=image, return_tensors="pt")

with torch.cpu.amp.autocast(dtype=torch.bfloat16):   # BF16 autocast on CPU
    for i in range(0,100000):
        outputs = model(**inputs)

pred = outputs.logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[pred])

