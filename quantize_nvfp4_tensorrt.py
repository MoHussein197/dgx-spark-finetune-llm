#!/usr/bin/env python3
"""
Merge LoRA adapter + Quantize to NVFP4 for inference
Run with: conda activate nvfp4 && python quantize_nvfp4.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
import os

# Configuration
BASE_MODEL = "HuggingFaceTB/SmolLM3-3B"
LORA_PATH = "./output/smollm3-3b-reasoning-lora"
MERGED_PATH = "./output/smollm3-3b-merged"
NVFP4_PATH = "./output/smollm3-3b-nvfp4"
CALIB_SIZE = 256


def merge_lora():
    """Merge LoRA adapter with base model"""
    print("=" * 60)
    print("[1/3] Merging LoRA adapter with base model...")
    print("=" * 60)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)

    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model = model.merge_and_unload()

    print(f"Saving merged model to {MERGED_PATH}...")
    os.makedirs(MERGED_PATH, exist_ok=True)
    model.save_pretrained(MERGED_PATH, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_PATH)

    print("Merged model saved!")
    return model, tokenizer


def quantize_nvfp4(model, tokenizer):
    """Quantize merged model to NVFP4"""
    print("\n" + "=" * 60)
    print("[2/3] Quantizing to NVFP4...")
    print("=" * 60)

    # Prepare calibration data
    print("Loading calibration dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    calib_data = []
    for i, sample in enumerate(dataset):
        if i >= CALIB_SIZE:
            break
        if sample["text"].strip():
            tokens = tokenizer(
                sample["text"],
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="max_length"
            )
            calib_data.append(tokens["input_ids"].to(model.device))

    print(f"Prepared {len(calib_data)} calibration samples")

    def forward_loop(model):
        print("Running calibration...")
        for i, batch in enumerate(calib_data):
            if i % 50 == 0:
                print(f"  Calibration progress: {i}/{len(calib_data)}")
            with torch.no_grad():
                model(batch)

    print("Applying NVFP4 quantization...")
    model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop)

    print(f"\nExporting NVFP4 model to {NVFP4_PATH}...")
    os.makedirs(NVFP4_PATH, exist_ok=True)

    with torch.inference_mode():
        export_hf_checkpoint(model, NVFP4_PATH)

    tokenizer.save_pretrained(NVFP4_PATH)
    print("NVFP4 model exported!")

    return model


def main():
    print("=" * 60)
    print("LoRA Merge + NVFP4 Quantization Pipeline")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Step 1: Merge LoRA
    if os.path.exists(MERGED_PATH):
        print(f"Merged model found at {MERGED_PATH}, loading...")
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH)
    else:
        model, tokenizer = merge_lora()

    # Step 2: Quantize to NVFP4
    quantize_nvfp4(model, tokenizer)

    # Done
    print("\n" + "=" * 60)
    print("[3/3] Complete!")
    print("=" * 60)
    print(f"\nNVFP4 model saved to: {NVFP4_PATH}")
    print("\nTo run inference:")
    print(f"""
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{NVFP4_PATH}", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{NVFP4_PATH}")

inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
""")


if __name__ == "__main__":
    main()
