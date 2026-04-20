# LORA IS ALL YOU NEED FOR SAFETY ALIGNMENT OF REASONING LLMS

> **Work in Progress**

This repository contains the data and code to reproduce the results from the paper [LoRA is All You Need for Safety Alignment of Reasoning LLMs](https://arxiv.org/abs/2507.17075)

Please check back later for updates.

## Overview

You can use the code in this repository to compare LoRA and full-model fine-tuning for performing safety alignment on reasoning LLMs. We find that LoRA achieves strong safety alignment without harming reasoning performance.

In addition, the code allows you to experiment with different LoRA configurations. Our findings show that:
- Rank-1 updates are sufficient to achieve the best balance between reasoning and safety.
- The up-projection layers are the most critical, and applying LoRA to them alone can yield even better results.
- Middle layers contribute most effectively to safety alignment, compared to early or late layers.

More results and analysis can be found in our [paper](https://arxiv.org/abs/2507.17075).

## 📦 Installation

The minimal required packages are listed in `environment.yml`. You can run `conda env create -f environment.yml` for easy setup with Conda.

## ⚙️ Running Experiments

Each experiment consists of the following steps:
1. Perform safety alignment fine-tuning using either full-model fine-tuning or LoRA.  
2. Evaluate the safety of the fine-tuned models and the base model.  
3. (Optional) Evaluate reasoning with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) installed separately via pip, then use the helper scripts in this repo for second-stage scoring.

### 🎯 Safety Alignment Fine-tuning

Training is performed with `train.py`. Intermediate Hugging Face `checkpoint-*` saves are **disabled** (`save_strategy="no"`); after training, the **final** adapter or model is written to `./finetuned_models/<model>/<run_name>/` via `trainer.save_model`.

#### Full-model finetuning
##### Standard

Here is an example:

```bash
model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
mode="full"
per_device_bs=2
epochs=5

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    --per_device_bs $per_device_bs \
    --model_name $model_name \
    --epochs $epochs \
    --mode $mode
```

##### Training with DeepSpeed ZeRO-3

Set up the DeepSpeed configuration JSON file as needed, and pass it to the command via `--ds_config`. Include the `--shard` flag. Below is an example of fine-tuning a 32B model using the example config file `ds_config_zero3_32b.json`:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
mode="full"
epochs=1
per_device_bs=1

CUDA_VISIBLE_DEVICES=0,1 deepspeed \
    train.py \
    --ds_config ds_config_zero3_32b.json \
    --model_name "$model_name" \
    --epochs "$epochs" \
    --mode "$mode" \
    --per_device_bs $per_device_bs \
    --shard
```

#### LoRA Finetuning

We set the LoRA configuration through the `--mode` argument.  
Here are a few options:

- `lora_qkvo_mlp_r{int}` — Apply LoRA to both attention and MLP layers, with the specified rank *r*.  
- `lora_mlp_r{int}` — Apply LoRA only to MLP layers.  
- `lora_{string}_only_r{int}` — Apply LoRA only to a specific submodule within the MLP. The `{string}` can be one of `up_proj`, `down_proj`, or `gate_proj`.  
- `lora_{string}_only_from{int}_to{int}_r{int}` — Similar to the above, but restricts LoRA to specific layer indices.  
- `full` — Full-model fine-tuning instead of LoRA.  
- You can also find other variations in the definition of `parse_config_string()` in `train.py`, which includes several LoRA regularization methods that we explored.

Below is an example of applying LoRA only to the up-projection layers with layer indices from 16 to 31, with r=1.

```bash
model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
per_device_bs=2
mode="lora_up_proj_only_from16_to31_r1"
epochs=10

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --per_device_bs $per_device_bs --model_name $model_name --mode $mode --epochs $epochs
```

After training, LoRA weights live directly under the run directory, for example:

`./finetuned_models/deepseek-ai_DeepSeek-R1-Distill-Qwen-14B/lora_up_proj_only_from16_to31_r1_epochs_10/`

Use that path as `--lora_path` for sampling and safety evaluation (not a `checkpoint-*` subfolder).

### 🛡️ Safety Evaluation

#### 1. Sampling responses

The first step is to sample responses from the model and save them using `sample_responses.py`.

For a **LoRA model**, you need to provide both:
- the path to the saved PEFT LoRA weights via `--lora_path`, and  
- the base model path via `--model_path`.
Use `--dataset_name` to pick the harmful-prompt benchmark (outputs are named per benchmark so you can keep both runs in the same folder):

- **`walledai/StrongREJECT`** or **`strongreject`**: shorter, more direct harmful prompts → `strongreject_responses.json`
- **`harmbench`** or **`walledai/HarmBench`**: gated HarmBench prompts on the Hub ([`walledai/HarmBench`](https://huggingface.co/datasets/walledai/HarmBench)) — run `hf auth login` (or set `HF_TOKEN`) and accept the dataset access terms on the website. Subsets: **`standard`** (default), **`contextual`** (context + prompt), **`copyright`** — pass `--harmbench_config`. Outputs are named per subset, e.g. `harmbench_standard_responses.json`. Split defaults to **`train`** (`--harmbench_split`).

For a **LoRA model**, responses are written next to `--lora_path` as above.

For a **non-LoRA model** (e.g., a full-model fine-tuned model or the base model itself), you only need to specify `--model_path`. The responses file is written under that path (or under `./finetuned_models/.../base` when sampling the raw base model by short name), using the same filenames as for LoRA.

Example for a single finished LoRA run:

```bash
dataset_name="walledai/StrongREJECT"
size="14B"
model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-$size"
lora_path="./finetuned_models/deepseek-ai_DeepSeek-R1-Distill-Qwen-$size/lora_up_proj_only_from16_to31_r1_epochs_10"
batch_size=4

CUDA_VISIBLE_DEVICES=0,1 python sample_responses.py \
    --lora_path $lora_path \
    --model_path $model_path \
    --dataset_name $dataset_name \
    --batch_size $batch_size
```

#### 2. Evaluating Responses

The second step is to use `evaluate_safety.py` to evaluate the sampled responses using a safety evaluator (here, `meta-llama/Llama-Guard-3-8B`).  
The evaluation results are saved next to the response file, with `_safety_eval.json` appended to the stem (for example `strongreject_responses_safety_eval.json` or `harmbench_standard_responses_safety_eval.json`).

```bash
dataset_name="walledai/StrongREJECT"
size="14B"
lora_path="./finetuned_models/deepseek-ai_DeepSeek-R1-Distill-Qwen-$size/lora_up_proj_only_from16_to31_r1_epochs_10"
response_file="${lora_path}/strongreject_responses.json"
batch_size=4

CUDA_VISIBLE_DEVICES=0,1 python evaluate_safety.py \
    --response_file $response_file \
    --batch_size $batch_size
```

### 🧠 Reasoning Evaluation (optional, external harness)

Reasoning benchmarks in the paper used [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) as a **separate** install (for example `pip install lm-eval`), not as a copy inside this repository.

Workflow:

1. **[LoRA only]** From this repo root, merge adapters into a full model:
   `python lora_conversion.py --base_model_path <base> --lora_model_path <run_dir>`
   This writes `<run_dir>_merged` next to the adapter folder. Use `--delete` with the same paths to remove the merged folder when done.

2. Run `lm_eval` (from your installed package) on the merged model, full fine-tune, or base model, with `--log_samples` and an `--output_path` of your choice.

3. Post-process outputs in that directory:
   - **GPQA**: `python mcq_metric_gpqa.py --directory_path <lm_eval_output_subdir> --task <task_name>`
   - **AIME**: `python math_metric_llm_eval_general.py --tensor_parallel_size <n> --directory_path <lm_eval_output_subdir> --task <task_name>`

The GPQA/AIME helpers were adapted from [Small-Model-Learnability-Gap](https://github.com/Small-Model-Gap/Small-Model-Learnability-Gap).

#### Coding Benchmarks

We adapted [EvalPlus](https://github.com/evalplus/evalplus) for **HumanEval** and **MBPP**.

The original implementation included a *response prefix* designed for earlier models that did not explicitly support intermediate thinking process. This prefix — for example, *“Below is a Python script with a self-contained function that
efficiently solves the problem and passes corresponding tests:”* — was prepended to model outputs during generation. We found that this disadvantages models good at thinking — including the base model and LoRA-fine-tuned models — since the forced prefix disrupts their expected output format (which should always begin with a thinking process before generating the final code).  
As a result, these models may skip the reasoning process entirely, leading to unreasonably low performance. Therefore, we remove the response prefix in our evaluation to make it compatible with thinking models. We will add the code for this part to the repository soon. 


## 📄 Citation
If you find this work useful, please cite:

```bibtex
@article{xue2025lora,
  title={LoRA is All You Need for Safety Alignment of Reasoning LLMs},
  author={Xue, Yihao and Mirzasoleiman, Baharan},
  journal={arXiv preprint arXiv:2507.17075},
  year={2025}
}
```
