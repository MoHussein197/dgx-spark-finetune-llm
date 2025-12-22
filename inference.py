#!/usr/bin/env python3
"""
Inference and Export script for fine-tuned SmolLM3-3B with LoRA

Backends:
  - fp4: bitsandbytes FP4 (default, any GPU)
  - nvfp4-te: Transformer Engine NVFP4 (Blackwell GPUs)

Export options:
  - --merge: Merge LoRA adapter with base model and save
  - --export-nvfp4: Export merged model to NVFP4 format (TensorRT-LLM compatible)

Usage:
  python inference.py --prompt "Hello"                    # FP4 inference
  python inference.py --backend nvfp4-te --prompt "Hi"    # NVFP4 inference
  python inference.py --merge                             # Merge LoRA + save
  python inference.py --export-nvfp4                      # Export to NVFP4
"""

import argparse
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
ADAPTER_PATH = "./output/smollm3-3b-reasoning-lora"
MERGED_PATH = "./output/smollm3-3b-merged"
NVFP4_PATH = "./output/smollm3-3b-nvfp4"


def load_model_fp4(adapter_path: str):
    """Load model with bitsandbytes FP4 quantization"""
    print(f"Loading base model with bitsandbytes FP4: {MODEL_ID}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    return model, tokenizer


def load_model_nvfp4_te(adapter_path: str):
    """Load model with Transformer Engine NVFP4"""
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import NVFP4BlockScaling
    except ImportError:
        raise ImportError(
            "Transformer Engine not installed. Use Docker:\n"
            "  docker run --gpus all -v $(pwd):/workspace nvcr.io/nvidia/pytorch:25.05-py3 \\\n"
            "    python /workspace/inference.py --backend nvfp4-te --prompt 'Hello'"
        )

    print(f"Loading base model with Transformer Engine NVFP4: {MODEL_ID}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    # Store recipe for inference
    model._nvfp4_recipe = NVFP4BlockScaling()
    model._te_module = te

    return model, tokenizer


def load_merged_model(merged_path: str):
    """Load a pre-merged model (no LoRA)"""
    print(f"Loading merged model: {merged_path}")

    model = AutoModelForCausalLM.from_pretrained(
        merged_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(merged_path)

    return model, tokenizer


def merge_and_save(adapter_path: str, output_path: str):
    """Merge LoRA adapter with base model and save"""
    print("=" * 60)
    print("Merging LoRA adapter with base model")
    print("=" * 60)

    print(f"\nLoading base model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)

    print("\nMerge complete!")
    return model, tokenizer


def export_nvfp4_tensorrt(merged_path: str, output_path: str, calib_size: int = 256):
    """Export merged model to NVFP4 format for TensorRT-LLM"""
    try:
        import modelopt.torch.quantization as mtq
        from modelopt.torch.export import export_hf_checkpoint
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "nvidia-modelopt not installed. Use the nvfp4 conda environment:\n"
            "  conda activate nvfp4\n"
            "  python inference.py --export-nvfp4"
        )

    print("=" * 60)
    print("Exporting to NVFP4 (TensorRT-LLM format)")
    print("=" * 60)

    # Load merged model
    if not os.path.exists(merged_path):
        print(f"Merged model not found at {merged_path}")
        print("Run with --merge first to create the merged model")
        return

    print(f"\nLoading merged model: {merged_path}")
    model = AutoModelForCausalLM.from_pretrained(
        merged_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(merged_path)

    # Prepare calibration data
    print("\nLoading calibration dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    calib_data = []
    for i, sample in enumerate(dataset):
        if i >= calib_size:
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
                print(f"  Progress: {i}/{len(calib_data)}")
            with torch.no_grad():
                model(batch)

    # Quantize
    print("\nApplying NVFP4 quantization...")
    model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop)

    # Export
    print(f"\nExporting to: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    with torch.inference_mode():
        export_hf_checkpoint(model, dtype=torch.bfloat16, export_dir=output_path)

    tokenizer.save_pretrained(output_path)

    print("\nExport complete!")
    print(f"NVFP4 model saved to: {output_path}")
    print("\nTo serve with TensorRT-LLM:")
    print(f"  trtllm-serve {output_path} --backend pytorch")


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_nvfp4_te: bool = False,
    enable_thinking: bool = True,
):
    """Generate a response for the given prompt

    Args:
        enable_thinking: If True, uses /think mode for extended reasoning.
                        If False, uses /no_think for direct answers.
    """
    # System prompt controls extended thinking mode
    # /think = enable extended reasoning (shows thought process)
    # /no_think = disable extended thinking (direct answers)
    thinking_flag = "/think" if enable_thinking else "/no_think"
    system_content = f"{thinking_flag}\nYou are a helpful AI assistant."

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Use NVFP4 autocast if Transformer Engine is enabled
    if use_nvfp4_te and hasattr(model, '_nvfp4_recipe'):
        te = model._te_module
        recipe = model._nvfp4_recipe
        with torch.no_grad(), te.autocast(recipe=recipe):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    else:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response


def main():
    parser = argparse.ArgumentParser(
        description="Inference and Export for fine-tuned SmolLM3-3B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference with bitsandbytes FP4
  python inference.py --prompt "Explain quantum computing"

  # Inference with Transformer Engine NVFP4 (requires Docker/TE)
  python inference.py --backend nvfp4-te --prompt "Hello"

  # Merge LoRA adapter with base model
  python inference.py --merge

  # Export to NVFP4 for TensorRT-LLM (requires merged model)
  python inference.py --export-nvfp4

  # Interactive mode
  python inference.py
        """
    )

    # Backend selection
    parser.add_argument(
        "--backend",
        type=str,
        choices=["fp4", "nvfp4-te"],
        default="fp4",
        help="Quantization backend: fp4 (bitsandbytes) or nvfp4-te (Transformer Engine)"
    )

    # Paths
    parser.add_argument("--adapter", type=str, default=ADAPTER_PATH, help="Path to LoRA adapter")
    parser.add_argument("--merged-path", type=str, default=MERGED_PATH, help="Path for merged model")
    parser.add_argument("--nvfp4-path", type=str, default=NVFP4_PATH, help="Path for NVFP4 export")

    # Export actions
    parser.add_argument("--merge", action="store_true", help="Merge LoRA with base model and save")
    parser.add_argument("--export-nvfp4", action="store_true", help="Export to NVFP4 (TensorRT-LLM)")

    # Generation params
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--no-think", action="store_true", help="Disable extended thinking (/no_think mode)")

    args = parser.parse_args()

    # Handle export actions
    if args.merge:
        merge_and_save(args.adapter, args.merged_path)
        return

    if args.export_nvfp4:
        # First check if merged model exists, if not, create it
        if not os.path.exists(args.merged_path):
            print("Merged model not found. Creating it first...\n")
            merge_and_save(args.adapter, args.merged_path)
            print()
        export_nvfp4_tensorrt(args.merged_path, args.nvfp4_path)
        return

    # Load model for inference
    print("=" * 60)
    print(f"Loading model for inference")
    print(f"Backend: {args.backend}")
    print("=" * 60)

    use_nvfp4_te = args.backend == "nvfp4-te"

    if use_nvfp4_te:
        model, tokenizer = load_model_nvfp4_te(args.adapter)
    else:
        model, tokenizer = load_model_fp4(args.adapter)

    # Check GPU info
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        mem_used = torch.cuda.memory_allocated() / 1e9
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {mem_used:.1f} / {mem_total:.1f} GB")

    enable_thinking = not args.no_think
    thinking_mode = "disabled (/no_think)" if args.no_think else "enabled (/think)"
    print(f"Extended thinking: {thinking_mode}")

    # Single prompt mode
    if args.prompt:
        print("\n" + "=" * 60)
        print("User:", args.prompt)
        print("=" * 60)
        response = generate_response(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            use_nvfp4_te=use_nvfp4_te,
            enable_thinking=enable_thinking,
        )
        print("\nAssistant:", response)
    else:
        # Interactive mode
        print("\n" + "=" * 60)
        print("Interactive mode - Type 'quit' to exit")
        print("Tip: Start prompt with '/think' or '/no_think' to toggle mode")
        print("=" * 60)

        while True:
            try:
                prompt = input("\nUser: ").strip()
            except EOFError:
                break

            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            if not prompt:
                continue

            # Allow toggling thinking mode per-message
            msg_thinking = enable_thinking
            if prompt.startswith("/think "):
                msg_thinking = True
                prompt = prompt[7:]
            elif prompt.startswith("/no_think "):
                msg_thinking = False
                prompt = prompt[10:]

            response = generate_response(
                model, tokenizer, prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                use_nvfp4_te=use_nvfp4_te,
                enable_thinking=msg_thinking,
            )
            print("\nAssistant:", response)


if __name__ == "__main__":
    main()
