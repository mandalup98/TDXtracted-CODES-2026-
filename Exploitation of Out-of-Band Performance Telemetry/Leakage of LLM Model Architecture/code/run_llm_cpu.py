#!/usr/bin/env python3
import argparse
import time
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_id: str, dtype: torch.dtype, trust_remote_code: bool):
    # Compatible with both newer (dtype) and older (torch_dtype) Transformers APIs
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
        )

def should_use_chat(tokenizer, model_id: str, chat_mode: str) -> bool:
    if chat_mode == "on":
        return True
    if chat_mode == "off":
        return False

    # auto:
    if getattr(tokenizer, "chat_template", None):
        return True

    mid = model_id.lower()
    return ("instruct" in mid) or ("chat" in mid)

def build_chat_input_text(tokenizer, prompt: str, system: str = "You are a helpful assistant.") -> str:
    """
    If tokenizer has a chat_template, use it.
    Otherwise, fall back to a simple text wrapper so --chat on works for base models too.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Fallback template for base models (GPT-Neo, GPT-2, OPT, Pythia base, etc.)
    return f"System: {system}\nUser: {prompt}\nAssistant:"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hugging Face model id (e.g., Qwen/Qwen2.5-0.5B)")
    ap.add_argument("--prompt", required=True, help="Prompt text")

    # Choose one:
    ap.add_argument("--iters", type=int, default=0, help="Number of inference iterations (e.g., 1000)")
    ap.add_argument("--minutes", type=int, default=0, help="Run for N minutes (e.g., 15)")

    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--greedy", action="store_true", help="Greedy decoding (no sampling)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--chat", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--trust_remote_code", action="store_true",
                    help="Enable for models that require custom code (some RWKV repos).")
    args = ap.parse_args()

    # Require exactly one mode
    if (args.iters > 0) == (args.minutes > 0):
        raise SystemExit("Provide exactly one: --iters N  OR  --minutes M")

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    device = "cpu"
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = load_model(args.model, dtype=dtype, trust_remote_code=args.trust_remote_code).to(device)
    model.eval()

    use_chat = should_use_chat(tokenizer, args.model, args.chat)

    if use_chat:
        text = build_chat_input_text(tokenizer, args.prompt)
        inputs = tokenizer(text, return_tensors="pt")
    else:
        inputs = tokenizer(args.prompt, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    if args.greedy:
        gen_kwargs.update(do_sample=False)
    else:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)

    last = None
    print("Start Inference")
    with torch.inference_mode():
        if args.iters > 0:
            for _ in range(args.iters):
                last = model.generate(**inputs, **gen_kwargs)
        else:
            end_t = time.time() + (args.minutes * 60)
            while time.time() < end_t:
                last = model.generate(**inputs, **gen_kwargs)

    # Print only last output
    if use_chat:
        gen_only = last[0][inputs["input_ids"].shape[-1]:]
        print(tokenizer.decode(gen_only, skip_special_tokens=True))
    else:
        print(tokenizer.decode(last[0], skip_special_tokens=True))
    #os.system("shutdown -h now")

if __name__ == "__main__":
    main()

