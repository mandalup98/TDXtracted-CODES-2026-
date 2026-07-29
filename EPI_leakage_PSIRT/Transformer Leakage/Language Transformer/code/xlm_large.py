import time
import torch
import intel_extension_for_pytorch as ipex
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load tokenizer and model
model_name = "xlm-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Convert model to BF16 and optimize with AMX
model = model.to(torch.bfloat16)
model = ipex.optimize(model, dtype=torch.bfloat16)
model.eval()

# Prepare static input text
text = "I'm feeling <mask> today."
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to("cpu") for k, v in inputs.items()}  # AMX runs on CPU

# Find mask token index
mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

iteration = 0

with torch.no_grad():
    while True:
        # start = time.perf_counter()

        # Run inference
        with torch.cpu.amp.autocast(dtype=torch.bfloat16):
            outputs = model(**inputs)
            logits = outputs.logits

        # end = time.perf_counter()
        # latency_ns = (end - start) * 1e9  # convert seconds → nanoseconds

        # print(f"{int(latency_ns)}")  # print as integer nanoseconds

        iteration += 1
