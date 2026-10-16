"""Validate report examples, or train a local Phi-3 QLoRA adapter on a CUDA GPU."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEADINGS = ("## Summary", "## Findings", "## Next steps")
NOT_FOUND = "I cannot find this information in the uploaded document."


def load_examples(path):
    examples = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        label = f"{path}:{number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label}: invalid JSON: {error.msg}") from error
        messages = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"{label}: expected system, user, and assistant messages.")
        for message, role in zip(messages, ("system", "user", "assistant")):
            if not isinstance(message, dict) or message.get("role") != role:
                raise ValueError(f"{label}: expected a {role} message.")
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ValueError(f"{label}: every message needs nonempty text content.")
        answer = messages[-1]["content"].strip()
        if answer != NOT_FOUND:
            headings = [line for line in answer.splitlines() if line.startswith("## ")]
            if headings != list(HEADINGS):
                raise ValueError(f"{label}: report must contain Summary, Findings, Next steps, in order.")
            if "[1]" not in answer:
                raise ValueError(f"{label}: report needs a source citation.")
        examples.append({"messages": messages})
    if not examples:
        raise ValueError(f"{path}: no examples found.")
    return examples


def validate_splits(train, evaluation):
    def prompts(rows):
        return [json.dumps(row["messages"][:-1], sort_keys=True) for row in rows]

    train_prompts, eval_prompts = prompts(train), prompts(evaluation)
    if len(set(train_prompts)) != len(train_prompts) or len(set(eval_prompts)) != len(eval_prompts):
        raise ValueError("Duplicate prompts found within a dataset split.")
    if set(train_prompts) & set(eval_prompts):
        raise ValueError("Training and evaluation prompts overlap; use separate examples.")


def require_training_environment():
    packages = ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "datasets", "trl")
    missing = [package for package in packages if importlib.util.find_spec(package) is None]
    if missing:
        raise RuntimeError("Missing training packages: " + ", ".join(missing)
                           + ". Use a separate environment and training/requirements.txt.")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This QLoRA recipe requires a CUDA-enabled NVIDIA GPU and CUDA PyTorch. "
                           "Dataset validation works on this machine; Phi-3 training was not started.")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    return torch


def train_adapter(args, train, evaluation):
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch = require_training_environment()
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Choose a new --output.")
    set_seed(42)
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    if not tokenizer.chat_template:
        raise ValueError("The model tokenizer must provide a chat template.")
    # Phi-3 ends each chat turn with <|end|>, rather than <|endoftext|>.
    if args.model == "microsoft/Phi-3-mini-4k-instruct":
        tokenizer.eos_token = "<|end|>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def dataset(rows):
        prepared = []
        for row in rows:
            messages = row["messages"]
            tokens = tokenizer.apply_chat_template(messages, tokenize=True)
            if len(tokens) > args.max_length:
                raise ValueError(f"An example uses {len(tokens)} tokens, exceeding --max-length "
                                 f"{args.max_length}. Shorten it or raise the limit; targets are not truncated.")
            prepared.append({"prompt": messages[:-1], "completion": messages[-1:]})
        return Dataset.from_list(prepared)

    train_data, eval_data = dataset(train), dataset(evaluation)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        torch_dtype=dtype,
        attn_implementation="eager",
        device_map={"": torch.cuda.current_device()},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        ),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=train_data, eval_dataset=eval_data,
        args=SFTConfig(
            output_dir=str(output), max_steps=args.max_steps,
            max_length=args.max_length, per_device_train_batch_size=1,
            per_device_eval_batch_size=1, gradient_accumulation_steps=4,
            learning_rate=2e-4, warmup_ratio=0.1,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=use_bf16, fp16=not use_bf16, optim="adamw_torch",
            completion_only_loss=True, packing=False,
            logging_steps=1, eval_strategy="no", save_strategy="no",
            report_to="none", seed=42, data_seed=42, dataloader_num_workers=0,
        ),
    )
    baseline = trainer.evaluate()
    result = trainer.train()
    evaluation_metrics = trainer.evaluate()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    manifest = {
        "base_model": args.model, "train_examples": len(train), "eval_examples": len(evaluation),
        "max_steps": args.max_steps, "max_length": args.max_length,
        "baseline": baseline, "training": result.metrics, "evaluation": evaluation_metrics,
        "note": "Demonstration adapter. Compare generated reports before deploying; loss alone is insufficient.",
    }
    (output / "experiment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Adapter and evaluation metrics saved to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=HERE / "data" / "report_train.jsonl")
    parser.add_argument("--eval", type=Path, default=HERE / "data" / "report_eval.jsonl")
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--output", type=Path, default=HERE / "output" / "report-phi3")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--validate-only", action="store_true", help="Check JSONL locally without training packages or downloads.")
    parser.add_argument("--check-env", action="store_true", help="Check dependencies and CUDA without downloading model weights.")
    parser.add_argument("--local-files-only", action="store_true", help="Only use Hugging Face weights already cached on this machine.")
    args = parser.parse_args()
    try:
        if args.max_steps < 1 or args.max_length < 64:
            raise ValueError("Use positive --max-steps and --max-length of at least 64.")
        train, evaluation = load_examples(args.train), load_examples(args.eval)
        validate_splits(train, evaluation)
        print(f"Validated {len(train)} training and {len(evaluation)} evaluation examples; no prompt overlap.", flush=True)
        if args.validate_only:
            print("Validation only: no model downloaded and no training performed.")
        elif args.check_env:
            require_training_environment()
        else:
            train_adapter(args, train, evaluation)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
