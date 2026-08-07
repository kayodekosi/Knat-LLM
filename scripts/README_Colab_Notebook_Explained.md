# Understanding the Generated Colab Notebook

A section-by-section, cell-by-cell walkthrough of what
`Knatware_LLM_FineTuning_V3_Colab_Template.ipynb` actually does, and why each
piece is there. This explains the **code**, not the n8n workflow that fills
it in — see [README_Notebook_Generator.md](README_Notebook_Generator.md) for
that side.

Every `{{TOKEN}}` shown below is exactly what appears in the template before
the n8n workflow substitutes real values. If you generated a notebook through
the form, your copy has real values in their place — the explanations still
apply the same way.

---

## Table of contents

1. [Environment Setup](#1-environment-setup)
2. [Hugging Face & Weights & Biases Login](#2-hugging-face--weights--biases-login)
3. [Core Configuration](#3-core-configuration)
4. [Load & Preprocess the Dataset](#4-load--preprocess-the-dataset)
5. [Load Tokenizer & Base Model](#5-load-tokenizer--base-model)
6. [Configure LoRA](#6-configure-lora-parameter-efficient-fine-tuning)
7. [Training & Hyperparameter-Tuning Arguments](#7-training--hyperparameter-tuning-arguments)
8. [Build the Trainer and Start Fine-Tuning](#8-build-the-trainer-and-start-fine-tuning)
9. [Save the Fine-Tuned Model](#9-save-the-fine-tuned-model)
10. [Push to the Hugging Face Hub](#10-optional-push-the-fine-tuned-model-to-the-hugging-face-hub)
11. [Quick Inference Test](#11-quick-inference-test)
12. [Hyperparameter Tuning Sweep](#12-optional-simple-hyperparameter-tuning-sweep)
13. [Temporary Browser Preview](#13-optional-quick-temporary-browser-preview)
14. [Permanent Deployment to Hugging Face Spaces](#14-deploy-permanently-to-a-free-browser-host-hugging-face-spaces)

---

## 1. Environment Setup

**What it's for:** confirms a GPU is actually attached to this Colab runtime,
then installs every Python package the rest of the notebook needs.

```python
!nvidia-smi
```

The leading `!` runs a shell command instead of Python — this is a Colab/
Jupyter convention throughout the notebook. `nvidia-smi` is NVIDIA's
diagnostic tool; if it errors or shows no GPU, the fix is in the comment
right above it: `Runtime > Change runtime type > Hardware accelerator > GPU`.
Nothing past this point will work without a GPU attached — fine-tuning even a
small model on CPU alone would take an impractical amount of time.

```python
!pip install -q -U transformers datasets accelerate peft trl bitsandbytes evaluate wandb sentencepiece
```

`-q` keeps the install output quiet; `-U` upgrades to the latest compatible
version even if something's already installed. Each package's job is spelled
out in the comment directly above this line in the notebook itself — briefly:
`transformers` loads the model/tokenizer, `datasets` loads training data,
`peft` implements LoRA, `trl` provides `SFTTrainer` (the actual training
loop), and `bitsandbytes` enables the 4-bit quantization that makes training
large models feasible on a free GPU.

```python
import os
import torch
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device:", torch.cuda.get_device_name(0))
```

Standard imports, plus a second, redundant GPU check — this time from inside
Python (`torch.cuda.is_available()`) rather than the shell, confirming
PyTorch itself can see the GPU `nvidia-smi` found. If `nvidia-smi` succeeded
but this prints `False`, the runtime type is set correctly but the installed
PyTorch build doesn't have CUDA support — reinstalling `torch` typically
resolves it.

---

## 2. (Optional) Hugging Face & Weights & Biases Login

**What it's for:** authenticates with the Hugging Face Hub (needed for gated
models or to push your results) and, optionally, Weights & Biases (a
dashboard for watching training metrics live).

```python
from huggingface_hub import login

HF_TOKEN = "{{HF_TOKEN}}"

if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("No Hugging Face token was configured — skipping login. ...")
```

`login(token=...)` stores the token for the rest of the session, so every
later Hugging Face API call (downloading a gated model, pushing a model,
creating a Space) is authenticated automatically. The `if HF_TOKEN:` guard
means an empty token doesn't crash the cell — it just skips login and prints
a reminder. Where `{{HF_TOKEN}}` actually comes from (and why it's usually
blank by default) is covered in the "Configuration: your Hugging Face token"
section of `README_Notebook_Generator.md`.

```python
import wandb
# wandb.login()  # commented out
os.environ["WANDB_DISABLED"] = "true"
```

[Weights & Biases](https://wandb.ai) is an optional experiment-tracking
service — useful if you want live loss curves in a browser dashboard instead
of just printed numbers. It's off by default (`WANDB_DISABLED = "true"`);
uncommenting `wandb.login()` and removing that line turns it on.

---

## 3. Core Configuration

**What it's for:** every value that controls *what* gets trained — the
model, the dataset, and where output is written — lives in one cell so
there's a single place to look.

```python
MODEL_NAME = "{{MODEL_NAME}}"
DATASET_NAME = "{{DATASET_NAME}}"
DATASET_SPLIT = "{{DATASET_SPLIT}}"
DATASET_TEXT_FIELD = "{{DATASET_TEXT_FIELD}}"
OUTPUT_DIR = "{{OUTPUT_DIR}}"
MAX_SEQ_LENGTH = {{MAX_SEQ_LENGTH}}
USE_4BIT = {{USE_4BIT}}
```

- `MODEL_NAME` / `DATASET_NAME` — Hugging Face Hub repository ids (e.g.
  `"Qwen/Qwen2.5-1.5B-Instruct"`), resolved by `AutoModelForCausalLM` and
  `load_dataset` later in the notebook to download the right files.
- `DATASET_SPLIT` — datasets aren't always split into `"train"` /
  `"validation"` / `"test"`; some use different names entirely. This value is
  auto-detected against the dataset's real available splits when the
  notebook was generated, specifically so the next cell doesn't crash trying
  to load a split that doesn't exist.
- `DATASET_TEXT_FIELD` — which column in the dataset holds the actual text to
  train on. Also auto-detected/verified against the dataset's real columns.
- `OUTPUT_DIR` — a local folder path (created automatically) where model
  checkpoints and the final result are written during this Colab session.
  This is *ephemeral storage* — it disappears when the runtime disconnects,
  which is why Sections 9–10 and 14 exist to save the result somewhere
  permanent (the Hugging Face Hub).
- `MAX_SEQ_LENGTH` — the maximum number of tokens processed per training
  example. Longer sequences need proportionally more GPU memory; this value
  is capped (both in the n8n workflow and implicitly here) to something that
  fits comfortably on a free-tier GPU.
- `USE_4BIT` — whether the model's weights are loaded in 4-bit precision
  (a technique called **QLoRA**). This is what makes it possible to fine-tune
  models that would otherwise be too large to fit in a free GPU's memory —
  the tradeoff is a small amount of numerical precision, which in practice
  barely affects fine-tuning quality.

The `print(...)` lines at the end of this cell aren't functional — they just
echo the configuration back so you can visually confirm it before running
anything expensive.

---

## 4. Load & Preprocess the Dataset

**What it's for:** downloads the training data and gets it into the exact
shape `SFTTrainer` expects: a single text column per example.

```python
dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

print(dataset)
print("\nSample record:")
print(dataset[0])
```

`load_dataset` is the `datasets` library's universal loader — for a Hub
dataset id, it downloads and caches the data automatically. `dataset[0]`
prints the very first example so you can eyeball the actual data structure
before training starts.

```python
if DATASET_TEXT_FIELD not in dataset.column_names:
    raise ValueError(
        f"DATASET_TEXT_FIELD is set to '{DATASET_TEXT_FIELD}', but the dataset's actual "
        f"columns are: {dataset.column_names}. ..."
    )
```

This check exists because a mismatched text-field name is one of the most
common ways this kind of notebook breaks — and if left unchecked, the failure
wouldn't surface until several cells later, deep inside `SFTTrainer`, with a
much less obvious error message. Failing immediately, with the dataset's real
column names printed out, turns a confusing debugging session into a
one-line fix.

```python
# def formatting_func(example):
#     return f"""### Instruction:
# {example['instruction']}
#
# ### Response:
# {example['response']}"""

formatting_func = None
```

Not every dataset already has one column containing a fully-formatted
training string — some split instructions and responses into separate
columns instead. `formatting_func`, if defined, is a function `SFTTrainer`
calls on every example to combine whatever columns exist into one training
string. It's commented out and defaults to `None` because most datasets
already have a usable single text column — uncomment and adapt the template
if yours doesn't.

---

## 5. Load Tokenizer & Base Model

**What it's for:** downloads the actual model weights and the matching
tokenizer (the component that converts text into the numeric token ids a
model actually operates on).

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"
```

`AutoTokenizer` inspects `MODEL_NAME` and loads whichever tokenizer
implementation that specific model actually uses (they differ by
architecture). `trust_remote_code=True` allows the Hub to run a small amount
of model-specific code some architectures need — standard practice for
`Auto*` classes on well-known public models. Many causal language models
don't define a padding token at all (they were never trained with one), so
this reuses the end-of-sequence token as a stand-in — a widely-used
convention.

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=USE_4BIT,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config if USE_4BIT else None,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

if USE_4BIT:
    model = prepare_model_for_kbit_training(model)
```

`BitsAndBytesConfig` describes *how* to quantize the model when `USE_4BIT` is
on: `"nf4"` is a quantization format designed specifically for neural network
weights (better accuracy than naive 4-bit rounding), `bnb_4bit_compute_dtype`
is the precision used for the actual math during training even though
weights are stored in 4 bits, and `bnb_4bit_use_double_quant` squeezes out a
little more memory savings by also quantizing the quantization constants
themselves. `device_map="auto"` lets `accelerate` figure out GPU placement
automatically — on a single-GPU Colab runtime this just means "put it on the
GPU." `use_cache = False` is required during training (it's an
inference-time optimization that conflicts with gradient checkpointing).
`prepare_model_for_kbit_training` applies several small adjustments
(stabilizing certain layers, enabling gradient flow through the quantized
weights) needed specifically for training a 4-bit model — it's a no-op with
no effect if `USE_4BIT` is off.

---

## 6. Configure LoRA (Parameter-Efficient Fine-Tuning)

**What it's for:** sets up **LoRA** — instead of updating all of a model's
weights (expensive, and usually impossible on a free GPU for anything but
the smallest models), LoRA freezes the original weights and trains small
additional "adapter" matrices alongside them. This is dramatically cheaper
while still meaningfully changing the model's behavior.

```python
LORA_R = {{LORA_R}}
LORA_ALPHA = {{LORA_ALPHA}}
LORA_DROPOUT = {{LORA_DROPOUT}}
LORA_TARGET_MODULES = {{LORA_TARGET_MODULES}}
```

- `LORA_R` (rank) — the size of the adapter matrices. Higher values give the
  adapter more capacity to learn, at the cost of more memory and slightly
  slower training. 8–64 is a typical range.
- `LORA_ALPHA` — a scaling factor applied to the adapter's output, commonly
  set to roughly twice `LORA_R`.
- `LORA_DROPOUT` — randomly zeroes out a fraction of adapter connections
  during training, a standard technique to reduce overfitting.
- `LORA_TARGET_MODULES` — *which* layers inside the model get an adapter
  attached. Every model architecture names its internal layers slightly
  differently (`q_proj`/`k_proj`/`v_proj`/`o_proj` for Llama-family models,
  `qkv_proj` for Phi-3, and so on) — this list has to match real layer names
  for the specific model being fine-tuned.

```python
_all_module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
_valid_targets = [m for m in LORA_TARGET_MODULES if m in _all_module_names]
if not _valid_targets:
    import torch.nn as nn
    _valid_targets = sorted({
        name.split(".")[-1] for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and "lm_head" not in name
    })
LORA_TARGET_MODULES = _valid_targets
```

`model.named_modules()` walks every layer inside the loaded model and
returns its full path (e.g. `"model.layers.0.self_attn.q_proj"`); taking
just the last segment after the final `.` gives the short layer name that
`target_modules` actually expects. This cell cross-checks the configured
target names against what's *really* inside this specific model, and — if
none of them match at all — automatically falls back to every `nn.Linear`
layer it can find (excluding the output/`lm_head` layer, which usually
shouldn't get a LoRA adapter). Without this check, a name mismatch would
otherwise surface as a fairly unhelpful error from `peft` itself.

```python
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

`LoraConfig` packages all of the above into the object `peft` expects.
`bias="none"` means bias terms aren't separately trained (the standard,
memory-efficient default). `task_type="CAUSAL_LM"` tells `peft` this is a
next-token-prediction model, which affects how it wraps certain layers.
`get_peft_model` actually attaches the adapters to the loaded model — from
this point on, `model` is a wrapped version that only updates the adapter
weights during training. `print_trainable_parameters()` reports what
fraction of the model is actually being trained — with LoRA this is
typically under 1–2% of the total parameter count, which is the whole point:
almost all of the model stays frozen.

---

## 7. Training & Hyperparameter-Tuning Arguments

**What it's for:** the actual training recipe — how long to train, how much
data to process at once, how aggressively to update weights, and so on.

```python
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs={{NUM_TRAIN_EPOCHS}},
    per_device_train_batch_size={{PER_DEVICE_TRAIN_BATCH_SIZE}},
    gradient_accumulation_steps={{GRADIENT_ACCUMULATION_STEPS}},
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_steps=50,
    learning_rate={{LEARNING_RATE}},
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="{{LR_SCHEDULER_TYPE}}",
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    max_length=MAX_SEQ_LENGTH,
    dataset_text_field=DATASET_TEXT_FIELD,
    report_to="none",
)
```

Grouped by what each one actually controls:

- **How much data, how many times** — `num_train_epochs` (full passes over
  the dataset; 1–3 is typical for instruction fine-tuning, since more risks
  the model memorizing rather than generalizing), `per_device_train_batch_size`
  (examples processed together per step — the first thing to reduce if you
  hit a `CUDA out of memory` error), and `gradient_accumulation_steps`
  (accumulates gradients over several small batches before actually updating
  weights, simulating a larger effective batch size without the memory cost
  of an actually larger batch).
- **How aggressively weights update** — `learning_rate` (the single most
  impactful value to tune; too high and training becomes unstable, too low
  and it barely learns anything), `max_grad_norm` (clips extreme gradient
  values, a standard safeguard against destabilizing spikes),
  `warmup_ratio` (ramps the learning rate up gradually at the very start
  instead of applying it at full strength immediately), and
  `lr_scheduler_type` (the shape of the learning-rate curve after warmup —
  `"cosine"` smoothly decays it toward zero, which is generally a safe
  default).
- **Memory/speed tradeoffs** — `optim="paged_adamw_32bit"` is a memory-
  efficient optimizer variant designed to pair well with quantized models,
  `bf16=True` uses a reduced-precision numeric format supported by modern
  GPUs (faster, less memory, negligible accuracy impact), and
  `gradient_checkpointing=True` trades some compute time for memory by not
  storing every intermediate activation during the forward pass.
- **Bookkeeping** — `logging_steps`/`save_steps` control how often progress
  prints and checkpoints save, `max_length`/`dataset_text_field` tell
  `SFTTrainer` how to actually read the dataset (matching the values set in
  Section 3), and `report_to="none"` disables external experiment trackers
  by default (set to `"wandb"` if you logged in during Section 2).

---

## 8. Build the Trainer and Start Fine-Tuning

**What it's for:** wires everything from the sections above together and
runs the actual training loop.

```python
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_func,
    processing_class=tokenizer,
    peft_config=lora_config,
)
```

`SFTTrainer` (Supervised Fine-Tuning Trainer, from the `trl` library) is a
purpose-built wrapper around the lower-level training loop that handles
tokenization, batching, and the LoRA-specific plumbing automatically —
significantly less code than implementing the loop by hand. Every argument
here is a value already assembled in earlier sections.

```python
train_result = trainer.train()
print(train_result)
```

This is the long-running cell — actual training happens here, with progress
(loss, learning rate, step count) printing every `logging_steps` steps as
configured in Section 7. Depending on model size, dataset size, and epoch
count, this can take anywhere from a few minutes to a few hours.
`train_result` is a small object summarizing the run (final loss, total
steps, timing).

---

## 9. Save the Fine-Tuned Model

**What it's for:** writes the trained result to disk — remember, `OUTPUT_DIR`
is local, ephemeral storage inside this Colab session.

```python
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

Because LoRA only trains small adapter matrices, `save_pretrained` here
writes just the adapter weights — typically a few megabytes up to around
100MB, rather than the full multi-gigabyte base model.

```python
MERGE_AND_SAVE = {{MERGE_AND_SAVE}}

if MERGE_AND_SAVE:
    merged_model = trainer.model.merge_and_unload()
    merged_output_dir = OUTPUT_DIR + "-merged"
    merged_model.save_pretrained(merged_output_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_output_dir)
```

`merge_and_unload()` mathematically folds the LoRA adapter weights directly
into the base model's weights, producing one standalone model that doesn't
need the `peft` library to load or run — useful for deployment (Section 14
does exactly this if it wasn't already done here). This is optional here
because it roughly doubles the disk space used (a full copy of the model, at
full size, alongside the small adapter) and isn't needed if you only plan to
load the adapter on top of the base model later.

---

## 10. (Optional) Push the Fine-Tuned Model to the Hugging Face Hub

```python
PUSH_TO_HUB = {{PUSH_TO_HUB}}
HUB_REPO_ID = "{{HUB_REPO_ID}}"

if PUSH_TO_HUB:
    trainer.model.push_to_hub(HUB_REPO_ID)
    tokenizer.push_to_hub(HUB_REPO_ID)
```

`push_to_hub` uploads whatever's currently in memory (the LoRA adapter, in
this case — not the merged model) to a Hugging Face Hub repository, creating
it if it doesn't already exist. This requires having logged in with a token
that has write access back in Section 2. Once pushed, the adapter is
reloadable from anywhere with `PeftModel.from_pretrained(base_model,
HUB_REPO_ID)`, without needing this Colab session at all.

---

## 11. Quick Inference Test

**What it's for:** a fast sanity check that fine-tuning actually changed the
model's behavior, before investing time in deployment.

```python
inference_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config if USE_4BIT else None,
    device_map="auto",
    trust_remote_code=True,
)
inference_model = PeftModel.from_pretrained(inference_model, OUTPUT_DIR)

generator = pipeline("text-generation", model=inference_model, tokenizer=tokenizer)
```

This deliberately reloads the base model fresh and reattaches the saved
adapter from `OUTPUT_DIR`, rather than reusing the in-memory `model` from
training — a clean reload closer to how the model would actually be loaded
elsewhere, catching any save/load issues immediately. `pipeline(...)` is
`transformers`' high-level convenience wrapper that handles tokenization,
generation, and decoding in one call.

```python
TEST_PROMPT = "{{TEST_PROMPT}}"

output = generator(
    TEST_PROMPT,
    max_new_tokens={{TEST_MAX_NEW_TOKENS}},
    do_sample=True,
    temperature={{TEST_TEMPERATURE}},
    top_p={{TEST_TOP_P}},
)

print(output[0]["generated_text"])
```

`max_new_tokens` caps how much text gets generated. `do_sample=True` means
generation isn't purely deterministic — `temperature` controls how random
(higher = more varied/creative, lower = more predictable/repetitive), and
`top_p` (nucleus sampling) restricts sampling to only the most probable
next-token candidates, avoiding very low-probability, likely-nonsensical
choices, while `temperature` still adds variation within that restricted set.

---

## 12. (Optional) Simple Hyperparameter Tuning Sweep

**What it's for:** a minimal example of comparing a hyperparameter across
several short training runs, rather than committing to one value blind.

```python
RUN_HYPERPARAMETER_SWEEP = False

if RUN_HYPERPARAMETER_SWEEP:
    learning_rates_to_try = [1e-4, 2e-4, 3e-4]
    sweep_results = {}
    for lr in learning_rates_to_try:
        sweep_args = SFTConfig(..., max_steps=20, learning_rate=lr, ...)
        sweep_trainer = SFTTrainer(...)
        result = sweep_trainer.train()
        sweep_results[lr] = result.training_loss
```

Off by default (`RUN_HYPERPARAMETER_SWEEP = False`) since it retrains the
model multiple times. Each iteration builds a fresh `SFTConfig` capped at
`max_steps=20` (a short run purely for comparing loss trends, not a full
training run) and records the final loss per learning rate tried. This is
meant as a template to extend — sweeping other hyperparameters, using more
steps, or using a proper tuning library (`optuna`, W&B Sweeps) for anything
beyond a quick comparison, as the notes in the notebook itself mention.

---

## 13. (Optional) Quick Temporary Browser Preview

**What it's for:** a fast, throwaway way to interact with the fine-tuned
model through an actual chat interface, without setting up permanent
hosting.

```python
!pip install -q gradio
import gradio as gr

def chat_fn(message, history):
    prompt = f"### Instruction:\n{message}\n\n### Response:\n"
    result = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
    full_text = result[0]["generated_text"]
    return full_text[len(prompt):].strip()

demo = gr.ChatInterface(fn=chat_fn, title="Knatware LLM — Live Preview (temporary link)")
demo.launch(share=True)
```

[Gradio](https://gradio.app) builds a simple web UI from a few lines of
Python. `chat_fn` builds the same instruction/response-style prompt used
during training, generates a reply, and strips the echoed-back prompt off
the front of the output so only the new reply is returned (`generator`
returns the *entire* generated text, prompt included, by default).
`share=True` is what makes this genuinely useful inside Colab — it tunnels
the locally-running Gradio server to a temporary public URL (something like
`https://xxxxxxxx.gradio.live`), so the interface can be opened from any
browser, on any device, while this cell keeps running. That link stops
working once the Colab runtime disconnects — Section 14 covers a permanent
alternative.

---

## 14. Deploy Permanently to a Free Browser Host (Hugging Face Spaces)

**What it's for:** publishes the model behind a stable, permanent URL that
keeps working after this Colab notebook is closed — [Hugging Face
Spaces](https://huggingface.co/spaces) is a free hosting platform purpose-
built for exactly this.

### 14.1 Deployment configuration

```python
HF_USERNAME = "{{HF_USERNAME}}"
SPACE_NAME = "{{SPACE_NAME}}"
DEPLOY_MODEL_REPO_ID = "{{DEPLOY_MODEL_REPO_ID}}"
SPACE_HARDWARE = "{{SPACE_HARDWARE}}"
SPACE_PUBLIC = {{SPACE_PUBLIC}}
```

Straightforward configuration values: which account the Space is created
under, what it's called (the final URL is
`huggingface.co/spaces/<HF_USERNAME>/<SPACE_NAME>`), which Hub repo holds the
model the Space will serve, which (free or paid) hardware tier to run it on,
and whether it's publicly visible.

### 14.2 Merge the LoRA adapter (if not already merged)

```python
if "merged_model" not in globals():
    merged_model = trainer.model.merge_and_unload()
    merged_output_dir = OUTPUT_DIR + "-merged"
    merged_model.save_pretrained(merged_output_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_output_dir)
else:
    print("Reusing already-merged model from Section 9.")
```

A Space shouldn't need the `peft` library just to combine a base model with
an adapter at inference time — merging once here (or reusing the merge from
Section 9, if `MERGE_AND_SAVE` was already set to `True` there) produces one
standalone model to deploy. `"merged_model" not in globals()` is a simple way
to check whether that variable already exists in this session, avoiding a
redundant merge.

### 14.3 Push the merged model

```python
merged_model.push_to_hub(DEPLOY_MODEL_REPO_ID, private=not SPACE_PUBLIC)
tokenizer.push_to_hub(DEPLOY_MODEL_REPO_ID, private=not SPACE_PUBLIC)
```

Uploads the full merged model (not just an adapter this time) to its own Hub
repository — separate from `HUB_REPO_ID` in Section 10, since that one holds
just the adapter. `private=not SPACE_PUBLIC` ties the model's visibility to
the same public/private choice made for the Space itself.

### 14.4 Write the Gradio app and requirements file

```python
%%writefile app.py
...
```

`%%writefile` is a Jupyter "cell magic" — instead of running as Python, the
entire cell's content is written verbatim to a file (`app.py` here). This is
how the notebook produces the actual application code a Hugging Face Space
will run. Inside that generated file:

```python
MODEL_REPO_ID = os.environ.get("MODEL_REPO_ID", "REPLACE_WITH_YOUR_MODEL_REPO_ID")
```

The Space reads which model to load from an environment variable rather than
having it hardcoded — set as a Space *secret* in Section 14.5 — specifically
so retraining and pushing a new model version under the same repo id doesn't
require touching or re-uploading `app.py` at all.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
```

Free-tier Spaces run on CPU only; this makes the app work correctly either
way, using faster GPU + `bfloat16` when available (e.g. on a paid Space tier)
and falling back to CPU + `float32` otherwise.

The rest of `app.py` builds a `gr.Blocks` interface (Gradio's more
customizable layout API, versus the simpler `ChatInterface` used in Section
13) with adjustable generation settings (max tokens, temperature, top-p)
exposed as sliders, and a chat function structurally identical to Section
13's `chat_fn`.

```python
%%writefile requirements.txt
transformers>=4.44.0
torch>=2.2.0
gradio>=4.40.0
accelerate>=0.33.0
sentencepiece
safetensors
```

The Space installs exactly these packages (and nothing else) before running
`app.py` — this is the complete list of what that file's imports actually
need, independent of whatever's installed in this Colab session.

### 14.5 Create the Space and upload the files

```python
from huggingface_hub import HfApi
api = HfApi()

space_repo_id = f"{HF_USERNAME}/{SPACE_NAME}"

api.create_repo(
    repo_id=space_repo_id,
    repo_type="space",
    space_sdk="gradio",
    private=not SPACE_PUBLIC,
    exist_ok=True,
)

api.add_space_secret(repo_id=space_repo_id, key="MODEL_REPO_ID", value=DEPLOY_MODEL_REPO_ID)

api.upload_file(path_or_fileobj="app.py", path_in_repo="app.py", repo_id=space_repo_id, repo_type="space")
api.upload_file(path_or_fileobj="requirements.txt", path_in_repo="requirements.txt", repo_id=space_repo_id, repo_type="space")
```

`HfApi` is a general-purpose client for Hugging Face Hub operations beyond
what `push_to_hub` covers. `create_repo(..., repo_type="space",
space_sdk="gradio")` provisions a new Space configured to run a Gradio app;
`exist_ok=True` means re-running this cell (say, after a code change) reuses
the existing Space instead of erroring out. `add_space_secret` is what wires
up the environment variable `app.py` reads for `MODEL_REPO_ID` — Space
secrets aren't visible in the Space's public files, which is the appropriate
way to pass this kind of configuration in. The two `upload_file` calls push
the two files written in 14.4; Hugging Face detects the push automatically,
builds a container from `requirements.txt`, and starts `app.py` — no manual
server setup involved.

### 14.6 Notes

Plain-text guidance in the notebook itself: expect a few minutes of build
time on first deploy, expect free `cpu-basic` Spaces to "sleep" after
inactivity (a brief delay on the next visit is normal), how to publish a
retrained model later (re-run 9, 14.2, and 14.3 with the same
`DEPLOY_MODEL_REPO_ID`), how to embed a Space elsewhere via `<iframe>`, and
where to look if replies feel slow (upgrading `SPACE_HARDWARE`, or switching
`app.py` to `llama-cpp-python` with a GGUF-converted model for faster CPU
inference).

---

## A note on the `{{PLACEHOLDER}}` tokens themselves

Every `{{TOKEN}}` seen throughout this document follows the same pattern:
it's plain text inside a Python string or a bare value, substituted by the
n8n workflow before the file is saved as a `.ipynb`. A cell like

```python
MODEL_NAME = "{{MODEL_NAME}}"
```

becomes, after generation,

```python
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
```

— ordinary Python, nothing left to fill in. If you ever want to fill a
template copy in by hand instead of using the form, that's all "filling in a
placeholder" means: find `{{TOKEN}}`, replace it (keeping the surrounding
quotes for string values, dropping them for numeric/boolean ones like
`MAX_SEQ_LENGTH` or `USE_4BIT`).

---

## License / attribution

© Knatware Technology — developed by Kayode Okosi, LLM Developer.
