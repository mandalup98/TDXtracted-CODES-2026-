import torch, intel_extension_for_pytorch as ipex
from transformers import BeitFeatureExtractor, BeitForMaskedImageModeling
from PIL import Image
import requests
import os
os.environ["ONEDNN_VERBOSE"] = "0"

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-large-patch16-224-pt22k')
model = BeitForMaskedImageModeling.from_pretrained('microsoft/beit-large-patch16-224-pt22k').eval()

# Optional: better memory layout for conv/patch-embed
model = model.to(memory_format=torch.channels_last)

# Enable BF16 execution on CPU; oneDNN will pick AMX BF16 kernels on 4th-gen Xeon
model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)

# Prepare inputs
inputs = feature_extractor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"].to(memory_format=torch.channels_last)
bool_masked_pos = inputs.get("bool_masked_pos")  # may be absent; pass if present

with torch.no_grad(), torch.cpu.amp.autocast(dtype=torch.bfloat16):
    for i in range(0,100000):
        outputs = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)

logits = outputs.logits
print(logits.shape)  # (B, num_visual_tokens, vocab_size) for MIM heads

