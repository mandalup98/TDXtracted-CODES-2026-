import time
import torch
import intel_extension_for_pytorch as ipex
from transformers import AutoTokenizer, MarianMTModel
# Model & text
model_name = "Helsinki-NLP/opus-mt-en-fr"
text = "Hello, how are you?"

# Tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = MarianMTModel.from_pretrained(model_name)

# AMX / BF16 optimization (CPU)
model = model.to(torch.bfloat16)
model = ipex.optimize(model, dtype=torch.bfloat16)
model.eval()

# Tokenize (keep integer dtype) on CPU
encoded = tokenizer(text, return_tensors="pt")
encoded = {k: v.to("cpu") for k, v in encoded.items()}

with torch.no_grad():
    #for _ in range(1000):
        #start = time.perf_counter_ns()
        with torch.cpu.amp.autocast(dtype=torch.bfloat16):
            for i in range(100000):
        	    generated_tokens = model.generate(**encoded, max_length=60)
        #end = time.perf_counter_ns()
        #print(end - start)  # integer nanoseconds per inference

# Optional: verify one translation (not timed)
# translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
# print("\nInput:", text)
# print("Translation (EN→FR):", translation)

