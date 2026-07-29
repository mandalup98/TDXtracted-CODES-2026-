import torch
import intel_extension_for_pytorch as ipex
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import os
import time

os.environ["ONEDNN_VERBOSE"] = "0"

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').eval()

# IPEX optimize in BF16 (enables oneDNN + AMX path)
model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)

inputs = processor(images=image, return_tensors="pt")

with torch.cpu.amp.autocast(dtype=torch.bfloat16):   # autocast to BF16 on CPU
    for i in range(100000):
        #start = time.perf_counter_ns()
        outputs = model(**inputs)
        #elapsed_ns = time.perf_counter_ns() - start
        #print(elapsed_ns, "ns")


pred = outputs.logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[pred])

