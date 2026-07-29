import torch, intel_extension_for_pytorch as ipex
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification
from PIL import Image
import requests
import os
os.environ["ONEDNN_VERBOSE"] = "0"

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = MobileViTFeatureExtractor.from_pretrained("apple/mobilevit-small")
model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small").eval()

# Optional: channels_last can speed up conv-heavy nets on CPU
model = model.to(memory_format=torch.channels_last)

# IPEX optimize in BF16 so oneDNN dispatches AMX BF16 kernels
model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)

inputs = feature_extractor(images=image, return_tensors="pt")

with torch.no_grad(), torch.cpu.amp.autocast(dtype=torch.bfloat16):
    for i in range(0,100000):
        outputs = model(**inputs)

pred = outputs.logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[pred])

