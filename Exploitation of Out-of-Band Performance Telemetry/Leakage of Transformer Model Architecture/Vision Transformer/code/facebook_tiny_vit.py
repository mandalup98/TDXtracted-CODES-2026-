import torch, intel_extension_for_pytorch as ipex
from transformers import AutoFeatureExtractor, DeiTForImageClassificationWithTeacher
from PIL import Image
import requests
import os
os.environ["ONEDNN_VERBOSE"] = "0"

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

fx = AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')
model = DeiTForImageClassificationWithTeacher.from_pretrained(
    'facebook/deit-tiny-distilled-patch16-224'
).eval()

# (Nice-to-have for convs/patch-embed)
model = model.to(memory_format=torch.channels_last)

# Optimize for BF16 on CPU; oneDNN will choose AMX BF16 kernels on 4th-gen Xeon
model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)

inputs = fx(images=image, return_tensors="pt")

with torch.no_grad(), torch.cpu.amp.autocast(dtype=torch.bfloat16):
    for i in range(0,100000):
        logits = model(**inputs).logits

pred = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[pred])

