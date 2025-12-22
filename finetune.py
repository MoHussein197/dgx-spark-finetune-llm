#!/usr/bin/env python3
"""
Fine-tuning SmolLM3-3B with LoRA and FP4/NVFP4/MXFP4 quantization
Dataset: TeichAI/claude-4.5-opus-high-reasoning-250x

Supports multiple quantization backends:
  - bitsandbytes FP4 (default): Works on any GPU with CUDA
  - NVIDIA Transformer Engine NVFP4: Optimized for Blackwell GPUs
  - QAT MXFP4/NVFP4: Post-SFT quantization-aware training

Usage:
  python finetune.py                    # bitsandbytes FP4 (adapter only)
  python finetune.py --save-merged      # FP4 + save merged model
  python finetune.py --use-nvfp4-te     # Transformer Engine NVFP4
  python finetune.py --qat-mxfp4        # SFT + QAT with MXFP4
  python finetune.py --qat-nvfp4        # SFT + QAT with NVFP4

Outputs:
  ./output/smollm3-3b-reasoning-lora/   # LoRA adapter (~240MB)
  ./output/smollm3-3b-merged/           # Merged model (~6GB, with --save-merged)
"""

import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import os

# Configuration
MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
DATASET_ID = "TeichAI/claude-4.5-opus-high-reasoning-250x"
OUTPUT_DIR = "./output/smollm3-3b-reasoning-lora"
MAX_SEQ_LENGTH = 8192


def get_quantization_config():
    """Configure NVFP4 (FP4) quantization with bitsandbytes"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",  # NVFP4 format
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # Nested quantization for memory efficiency
    )


def get_lora_config():
    """Configure LoRA adapter for efficient fine-tuning"""
    return LoraConfig(
        r=64,  # LoRA rank
        lora_alpha=128,  # LoRA alpha scaling
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SmolLM3-3B with LoRA + FP4/NVFP4/MXFP4")

    # Model and dataset
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_ID,
        help=f"Model ID from HuggingFace (default: {MODEL_ID})"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_ID,
        help=f"Dataset ID from HuggingFace (default: {DATASET_ID})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for LoRA adapter (default: {OUTPUT_DIR})"
    )

    # Quantization backends
    parser.add_argument(
        "--use-fp4",
        action="store_true",
        help="Use bitsandbytes FP4 quantization (default, works on any GPU)"
    )
    parser.add_argument(
        "--use-nvfp4",
        action="store_true",
        help="Use Transformer Engine NVFP4 4-bit (requires Blackwell GPU + Docker)"
    )
    parser.add_argument(
        "--use-mxfp8",
        action="store_true",
        help="Use Transformer Engine MXFP8 8-bit (requires Blackwell GPU + Docker)"
    )

    # Output options
    parser.add_argument(
        "--save-merged",
        action="store_true",
        help="Save merged model (adapter + base) after training"
    )
    parser.add_argument(
        "--merged-output",
        type=str,
        default="./output/smollm3-3b-merged",
        help="Output path for merged model"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine quantization backend
    use_te = args.use_nvfp4 or args.use_mxfp8
    use_bitsandbytes = not use_te  # Default to bitsandbytes if no TE flag

    # Import Transformer Engine if needed
    te = None
    te_recipe = None
    if use_te:
        try:
            import transformer_engine.pytorch as te
            import transformer_engine
            print(f"Transformer Engine version: {transformer_engine.__version__}")

            # Import the appropriate recipe
            if args.use_mxfp8:
                try:
                    from transformer_engine.common.recipe import MXFP8BlockScaling, Format
                    te_recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)
                    print("Using MXFP8BlockScaling recipe (8-bit)")
                except ImportError:
                    raise ImportError(
                        "MXFP8BlockScaling not available. Need Transformer Engine >= 2.9.0\n"
                        "Use container: nvcr.io/nvidia/pytorch:25.11-py3"
                    )
            else:  # NVFP4
                try:
                    from transformer_engine.common.recipe import NVFP4BlockScaling
                    te_recipe = NVFP4BlockScaling()
                    print("Using NVFP4BlockScaling recipe (4-bit)")
                except ImportError:
                    raise ImportError(
                        "NVFP4BlockScaling not available. Need Transformer Engine >= 2.9.0\n"
                        "Use container: nvcr.io/nvidia/pytorch:25.11-py3"
                    )

            quant_type = "MXFP8" if args.use_mxfp8 else "NVFP4"
            print(f"Transformer Engine {quant_type} enabled")
        except ImportError as e:
            raise ImportError(
                f"Transformer Engine error: {e}\n"
                "Use Docker: nvcr.io/nvidia/pytorch:25.11-py3"
            )

    # Determine backend name and output suffix
    if args.use_mxfp8:
        backend_name = "Transformer Engine MXFP8"
        output_suffix = "-mxfp8-lora"
    elif args.use_nvfp4:
        backend_name = "Transformer Engine NVFP4"
        output_suffix = "-nvfp4-lora"
    else:
        backend_name = "bitsandbytes FP4"
        output_suffix = "-lora"

    model_id = args.model
    dataset_id = args.dataset
    base_output_dir = args.output_dir

    # Adjust output dir based on backend
    if not base_output_dir.endswith(output_suffix):
        if "-lora" in base_output_dir:
            output_dir = base_output_dir.replace("-lora", output_suffix)
        else:
            output_dir = base_output_dir + output_suffix
    else:
        output_dir = base_output_dir

    print("=" * 60)
    print(f"Fine-tuning {model_id} with LoRA + {backend_name}")
    print("=" * 60)
    print(f"Dataset: {dataset_id}")
    print(f"Output: {output_dir}")

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute Capability: {cap[0]}.{cap[1]}")
        if use_te and cap[0] < 10:
            print("WARNING: NVFP4/MXFP4 with Transformer Engine is optimized for Blackwell GPUs (SM >= 100)")
    else:
        raise RuntimeError("CUDA not available. This script requires a GPU.")

    # Load dataset
    print("\n[1/5] Loading dataset...")
    dataset = load_dataset(dataset_id, split="train")
    print(f"Dataset loaded: {len(dataset)} examples")

    # Load tokenizer
    print("\n[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"\n[3/5] Loading model with {backend_name}...")

    if use_te:
        # For Transformer Engine: load in bf16, quantization happens during forward pass
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
    else:
        # For bitsandbytes: load with quantization config
        quantization_config = get_quantization_config()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    print("\n[4/5] Applying LoRA adapter...")
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training configuration
    print("\n[5/5] Setting up training...")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch = 8
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if not use_te else "adamw_torch",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        report_to="tensorboard",
        logging_dir=os.path.join(output_dir, "logs"),
        seed=42,
    )

    # Create trainer (with TE autocast if using Transformer Engine)
    if use_te:
        from transformer_engine.pytorch import fp8_autocast

        class TETrainer(SFTTrainer):
            """Custom trainer that wraps forward pass with Transformer Engine autocast"""
            def training_step(self, model, inputs, num_items_in_batch=None):
                with fp8_autocast(enabled=True, fp8_recipe=te_recipe):
                    return super().training_step(model, inputs, num_items_in_batch)

        trainer = TETrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

    # Start training
    print("\n" + "=" * 60)
    print(f"Starting training with {backend_name}...")
    print("=" * 60)

    trainer.train()

    # Save the final adapter
    print("\n" + "=" * 60)
    print("Saving LoRA adapter...")
    print("=" * 60)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nTraining complete! Adapter saved to: {output_dir}")

    # Save merged model (if requested)
    if args.save_merged:
        print("\n" + "=" * 60)
        print("Saving merged model (adapter + base)...")
        print("=" * 60)

        merged_model = model.merge_and_unload()
        os.makedirs(args.merged_output, exist_ok=True)
        merged_model.save_pretrained(args.merged_output, safe_serialization=True)
        tokenizer.save_pretrained(args.merged_output)
        print(f"Merged model saved to: {args.merged_output}")

    print("\n" + "=" * 60)
    print("All training complete!")
    print("=" * 60)
    print(f"\nAdapter: {output_dir}")
    if args.save_merged:
        print(f"Merged model: {args.merged_output}")
    print("\nTo use the fine-tuned model:")
    print(f"""
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("{model_id}")
model = PeftModel.from_pretrained(base_model, "{output_dir}")
tokenizer = AutoTokenizer.from_pretrained("{output_dir}")
""")


if __name__ == "__main__":
    main()
