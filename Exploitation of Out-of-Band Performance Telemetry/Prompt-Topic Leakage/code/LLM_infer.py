import os
import sys
import struct
import time

import torch
import intel_extension_for_pytorch as ipex
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_DIR = "./phi3_qa_lora"

MSR_PATH = "/dev/cpu/0/msr"
MSR_OFFSET = 0x610

START_VAL = 0x878D2100158AF0
STOP_VAL  = 0x878D2000158AF0


def write_msr(fd, value):
    os.pwrite(fd, struct.pack("<Q", value), MSR_OFFSET)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: sudo python3 {sys.argv[0]} '<prompt>'")
        sys.exit(1)

    prompt = sys.argv[1]

    torch.set_grad_enabled(False)
    torch.set_num_threads(os.cpu_count() or 1)

    fd = os.open(MSR_PATH, os.O_RDWR)

    try:
        tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=False)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            BASE,
            trust_remote_code=False,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        base.to("cpu")

        peft_model = PeftModel.from_pretrained(base, ADAPTER_DIR)

        # Merge LoRA into the base model so IPEX sees a plain model
        model = peft_model.merge_and_unload()
        model.eval()
        model.to("cpu")
        model.config.use_cache = True

        # Try IPEX LLM optimization for CPU AMX/BF16 path
        try:
            model = ipex.llm.optimize(
                model,
                dtype=torch.bfloat16,
                inplace=True,
                deployment_mode=True,
            )
            print("[INFO] IPEX LLM optimize enabled.")
        except Exception as e:
            print(f"[WARN] ipex.llm.optimize failed: {e}")
            print("[WARN] Continuing with plain CPU BF16 autocast.")

        # Use proper chat-template formatting so the model answers only the given prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        formatted_prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tok(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        for i in range(1):
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                out = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )

            
        for i in range(1):
            write_msr(fd, START_VAL)
            t0 = time.time()

            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                out = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )

            t1 = time.time()
            write_msr(fd, STOP_VAL)

            answer_ids = out[0][prompt_len:]
            answer = tok.decode(answer_ids, skip_special_tokens=True)

            print(f"\n=== Iteration {i + 1} ===")
            print(f"[TIME] {t1 - t0:.6f} sec")
            print(answer.strip())

            time.sleep(1)

    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
