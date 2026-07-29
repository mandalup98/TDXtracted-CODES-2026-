import torch, intel_extension_for_pytorch as ipex
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests
import os
os.environ["ONEDNN_VERBOSE"] = "0"

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224"
).eval()

# (Nice-to-have for Swin’s patch-embed/conv layers)
model = model.to(memory_format=torch.channels_last)

# Optimize for BF16 on CPU; oneDNN will pick AMX BF16 kernels on 4th-gen Xeon
model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)

inputs = processor(images=image, return_tensors="pt")
# channels_last helps only for 4D tensors; safe to keep inputs as-is

with torch.no_grad(), torch.cpu.amp.autocast(dtype=torch.bfloat16):
    for i in range(0,100000):
        outputs = model(**inputs)

logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

