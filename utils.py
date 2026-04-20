# utils.py
import argparse
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import tempfile
import shutil

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA fine-tuning")
    return parser.parse_args()

def create_optimizer_and_scheduler(model, learning_rate=5e-5, weight_decay=1e-4, num_warmup_steps=500, num_training_steps=5000):
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    return optimizer, scheduler

# === Apply tokenizer chat template ===
def format_prompt(tokenizer, prompt):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        raise RuntimeError(
            "The tokenizer does not support `apply_chat_template`. "
            "Make sure you are using a chat model with a defined chat template, "
            "or switch to a non-chat-style prompt format manually."
        )


def load_prompts(dataset_name, split, field_name, num_samples=None):
    dataset = load_dataset(dataset_name, split=split)

    if field_name not in dataset.column_names:
        raise ValueError(f"Field '{field_name}' not found in dataset columns: {dataset.column_names}")

    if num_samples is None or num_samples > len(dataset):
        num_samples = len(dataset)

    return dataset.select(range(num_samples))[field_name]


STRONGREJECT_DATASET = "walledai/StrongREJECT"
HARMBENCH_DATASET = "walledai/HarmBench"
HARMBENCH_CONFIGS = frozenset({"standard", "contextual", "copyright"})


def normalize_safety_benchmark_name(name):
    """Map CLI / shorthand names to internal slugs: strongreject | harmbench."""
    n = (name or "").strip().lower()
    if n in ("walledai/strongreject", "strongreject"):
        return "strongreject"
    if n in ("harmbench", "walledai/harmbench"):
        return "harmbench"
    if n in ("swiss-ai/harmbench", "swiss_ai/harmbench"):
        raise ValueError(
            "HarmBench sampling now uses the gated Hub dataset "
            f"{HARMBENCH_DATASET!r} (HF login + accepting dataset terms). "
            "Pass 'harmbench' or 'walledai/HarmBench' instead."
        )
    raise ValueError(
        f"Unknown safety benchmark dataset {name!r}. "
        f"Use {STRONGREJECT_DATASET!r} or 'strongreject' for StrongREJECT; "
        f"'harmbench' or 'walledai/HarmBench' for HarmBench."
    )


def safety_benchmark_response_filename(slug, harmbench_config=None):
    if slug == "strongreject":
        return "strongreject_responses.json"
    if slug == "harmbench":
        cfg = (harmbench_config or "standard").lower()
        return f"harmbench_{cfg}_responses.json"
    raise ValueError(f"unknown slug: {slug}")


def safety_benchmark_metadata_filename(slug, harmbench_config=None):
    return safety_benchmark_response_filename(slug, harmbench_config).replace(
        "_responses.json", "_sampling_params.json"
    )


def _walledai_harmbench_user_prompt(row, config_name):
    """Build one user message string per row for walledai/HarmBench configs."""
    cfg = (config_name or "").lower()
    prompt = (row.get("prompt") or "").strip()
    if cfg == "contextual":
        ctx = row.get("context")
        if ctx is not None and str(ctx).strip():
            return f"{str(ctx).strip()}\n\n{prompt}"
        return prompt
    if cfg in ("standard", "copyright"):
        return prompt
    raise ValueError(f"unsupported HarmBench config for prompt formatting: {config_name!r}")


def load_safety_benchmark_prompts(
    slug,
    *,
    harmbench_config="standard",
    harmbench_split="train",
    num_samples=None,
):
    """
    Load plain-text harmful user prompts for a supported safety benchmark.

    StrongREJECT: short/direct harmful prompts (walledai/StrongREJECT train split).
    HarmBench: gated walledai/HarmBench — configs standard | contextual | copyright
    (all ship a train split). contextual prepends context + blank lines + prompt when
    context is present.
    """
    if slug == "strongreject":
        return load_prompts(STRONGREJECT_DATASET, "train", "prompt", num_samples=num_samples)

    if slug == "harmbench":
        cfg = (harmbench_config or "standard").lower()
        if cfg not in HARMBENCH_CONFIGS:
            raise ValueError(
                f"Unknown HarmBench config {harmbench_config!r}. "
                f"Use one of: {', '.join(sorted(HARMBENCH_CONFIGS))}."
            )
        ds = load_dataset(HARMBENCH_DATASET, cfg, split=harmbench_split)
        if num_samples is None or num_samples > len(ds):
            num_samples = len(ds)
        subset = ds.select(range(num_samples))
        return [_walledai_harmbench_user_prompt(row, cfg) for row in subset]

    raise ValueError(f"unknown slug: {slug}")


def lora_merge(base_model_path, lora_model_path):
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto", device_map="cpu")

    # Load and apply the LoRA adapter
    # Force adapter weights to load on CPU to avoid touching CUDA during merge.
    model = PeftModel.from_pretrained(model, lora_model_path, device_map="cpu")

    # Merge LoRA weights into the base model
    model = model.merge_and_unload()

    return model


def lora_merge_and_load_vllm(base_model_path, lora_model_path, tensor_parallel_size):
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(dir='./')

    try:
        # Load and merge model
        model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto", device_map="cpu")
        # Force adapter weights to load on CPU to avoid CUDA device contention
        # (common on shared JupyterHub setups where stray processes can hold GPUs).
        model = PeftModel.from_pretrained(model, lora_model_path, device_map="cpu")
        model = model.merge_and_unload()

        # Save to temp directory
        model.save_pretrained(temp_dir)

        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(temp_dir)

        # Load with vLLM
        from vllm import LLM
        llm = LLM(model=temp_dir, dtype="auto", tensor_parallel_size=tensor_parallel_size)

    finally:
        # Delete the temporary model files
        shutil.rmtree(temp_dir)

    return llm