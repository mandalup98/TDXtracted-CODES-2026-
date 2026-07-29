import time
import torch
import intel_extension_for_pytorch as ipex
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load tokenizer and model
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# AMX / BF16 optimization (CPU target)
model = model.to(torch.bfloat16)
model = ipex.optimize(model, dtype=torch.bfloat16)
model.eval()

# Static input on CPU
text = "translate English to French: I love pizza!"
inputs = tokenizer(text, return_tensors="pt").input_ids.to("cpu")

with torch.no_grad():
    while True:
        # start = time.perf_counter_ns()

        # Generation under BF16 autocast (applies to internal forward passes)
        with torch.cpu.amp.autocast(dtype=torch.bfloat16):
            outputs = model.generate(inputs, max_length=50)

        # end = time.perf_counter_ns()
        # latency_ns = end - start

        # Print only the latency in integer nanoseconds (one per line)
        # print(latency_ns)

# Optional: if you want to see the actual translation once (not timed), uncomment:
# translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("\nInput:", text)
# print("Output:", translation)
