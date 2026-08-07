# Knatware LLM Notebook Generator — n8n Workflow

Turns a 4-step web form into a ready-to-run Google Colab notebook for fine-tuning
an open-source LLM with LoRA/QLoRA — pulling live model and dataset choices
straight from the Hugging Face Hub.

**File:** `Knatware_LLM_Notebook_Generator_n8n_workflow.json`
**Companion file:** `Knatware_LLM_FineTuning_V3_Colab_Template.ipynb` (must sit next to this
workflow in the same n8n environment — see [Setup](#setup))

---

## What it does

1. You open a web form and search Hugging Face for a base model.
2. You pick a model from live results (or type any model id).
3. The workflow looks at that model's real config and searches Hugging Face for
   relevant training datasets, ranked by relevance to the model.
4. You pick a dataset (or type any dataset id).
5. A final form shows recommended hyperparameters (LoRA rank, sequence length,
   4-bit loading, etc.) pre-filled as placeholders — override anything you like.
6. You choose how to receive the notebook:
   - **Download** the finished `.ipynb`, or
   - **Push it to Google Drive** and get a link that opens directly in Colab.

The output is a fully-substituted Colab notebook — no `{{PLACEHOLDER}}` tokens left —
covering environment setup, Hugging Face/W&B login, dataset loading, LoRA
configuration, training, saving/pushing to the Hub, inference testing, and optional
deployment to a free Hugging Face Space.

---

## Architecture

```
Step 1 (search) ──▶ Config (edit token here) ──▶ Search HF Models ──▶ Build Model Dropdown
                                                                            │
                                                                            ▼
                                                    Step 2 (pick model, or type custom)
                                                                            │
                                                                            ▼
                                                                   Resolve Model Choice
                                                                            │
                                                                            ▼
                                                    Fetch Model Info (Hugging Face)
                                                                            │
                                                                            ▼
                                                          Build Dataset Search Params
                                                            │                      │
                                                            ▼                      ▼
                                            Search HF Datasets by Family   Search HF Datasets by Task
                                                            │                      │
                                                            └────────┬─────────────┘
                                                                     ▼
                                                        Join Dataset Searches (Merge node)
                                                                     ▼
                                                          Build Dataset Dropdown
                                                                     │
                                                                     ▼
                                                  Step 3 (pick dataset, or type custom)
                                                                     │
                                                                     ▼
                                                          Resolve Dataset Choice
                                                                     │
                                                                     ▼
                                                     Fetch Dataset Info (Hugging Face)
                                                                     │
                                                                     ▼
                                                          Derive Recommendations
                                                                     │
                                                                     ▼
                                              Step 4 (hyperparameters + delivery choice)
                                                                     │
                                                                     ▼
                                                  Validate & Sanitize Parameters
                                                                     │
                                                                     ▼
                                                          Fill Notebook Template
                                                                     │
                                                                     ▼
                                                            Delivery Method? (IF)
                                                       ┌──────────────┴──────────────┐
                                                       ▼                              ▼
                                             Upload to Google Drive          Step 5b – Download
                                                       │
                                                       ▼
                                                Build Colab Link
                                                       │
                                                       ▼
                                             Step 5a – Open in Colab
```

26 nodes total, including two sticky notes with in-canvas setup instructions.

---

## Why the generated notebook shouldn't error out

Everything the form collects passes through a dedicated **Validate &
Sanitize Parameters** node before it ever reaches the notebook — it doesn't
just trust whatever was typed in. Specifically, it:

- **Checks `DATASET_TEXT_FIELD` against the dataset's real columns.** If it
  doesn't exist (typo, or the user typed a field from a different dataset),
  it's replaced with the auto-detected column instead of shipping a notebook
  that crashes the moment `SFTTrainer` looks for it.
- **Clamps `MAX_SEQ_LENGTH`** to both the chosen model's actual context
  window *and* a 2048-token ceiling — long enough for most instruction data,
  short enough to avoid a `CUDA out of memory` error on a free-tier T4, even
  if someone types an enormous number.
- **Falls back to a non-empty `LORA_TARGET_MODULES` list** if the field is
  left blank or emptied out, using the architecture-appropriate defaults from
  `Derive Recommendations`.
- **Clamps every numeric hyperparameter** (`LORA_R`, `LORA_ALPHA`,
  `LORA_DROPOUT`, `NUM_TRAIN_EPOCHS`, batch size, gradient accumulation,
  generation settings) to ranges that won't crash or silently hang — a
  non-numeric or out-of-range value is replaced with a safe default rather
  than passed straight through to `TrainingArguments`.
- **Rebuilds Hub repo ids and the Space name from valid parts** (lowercased,
  stripped of anything that isn't a letter/number/`.`/`_`/`-`), so a stray
  space or symbol can't produce a repo id the Hugging Face Hub API rejects at
  push time.

Every substitution is recorded in a `validationWarnings` array so nothing
happens silently if you're inspecting the workflow's execution data.

**The generated notebook itself also defends against drift** between
generation time and run time (e.g., you hand-edit a value, or a dataset's
schema changes before you get around to running the notebook):

- **Section 4** checks that `DATASET_TEXT_FIELD` is actually a column on the
  loaded dataset immediately after loading it, and raises a clear error
  naming the real available columns if not — instead of a cryptic failure
  several cells later inside `SFTTrainer`.
- **Section 6** filters `LORA_TARGET_MODULES` against the model's real
  module names before building `LoraConfig`, and automatically falls back to
  auto-detected linear-layer names if none of the configured targets exist
  on that particular model — `peft`'s own error message for a target-module
  mismatch is not very actionable, so this catches it earlier with a
  specific, useful message instead.

None of this guarantees a training run will succeed (a model can still be
too large for the assigned GPU, a dataset can still be malformed in ways
that aren't inspectable ahead of time, etc.) — but it eliminates the class of
errors caused by mismatched or malformed *parameters*, which is what this
workflow controls.

---

## Prerequisites

- A running n8n instance (self-hosted or cloud) with the **`@n8n/n8n-nodes-langchain`**
  community/core nodes available for the sub-workflow described in the companion
  [LLM Chain Inference workflow](README_LLM_Chain_Inference.md) — not required for
  *this* workflow, which only uses core n8n nodes (Form, HTTP Request, Code, IF,
  Google Drive, Merge).
- A Google account, **if** you plan to use the "push to Google Drive & open in
  Colab" delivery option.
- (Optional) A Hugging Face account and access token, pasted into the
  **Config (edit token here)** node — needed for gated models (Llama, Gemma,
  etc.) or to raise Hugging Face API rate limits. See
  [Configuration: your Hugging Face token](#configuration-your-hugging-face-token).
  Not something end users of the form need to provide.

---

## Configuration: your Hugging Face token

There's exactly **one place** to configure a Hugging Face token: the
**Config (edit token here)** node, right after the Step 1 trigger. Open it and
edit the `HF_TOKEN` constant near the top of its code:

```js
const HF_TOKEN = "";  // <-- paste your token between the quotes
```

That single value is used for two things:

1. **Authenticating this workflow's own Hugging Face API calls** (model
   search, model info, dataset search, dataset info) — optional, works fine
   blank too, just with Hugging Face's lower unauthenticated rate limit and
   no access to gated models.
2. **Auto-filling the `HF_TOKEN` login cell** in every notebook this workflow
   generates (Section 2, "Login to Hugging Face") — optional; if blank, the
   generated notebook just prints a message and expects whoever runs it to
   paste their own token in instead.

Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**If you plan to share or commit this workflow file** (e.g. to GitHub): a
filled-in token is saved in plain text right there in the node's code, which
means it's also in the exported workflow JSON. Keep a private copy with the
real token, and a separate blank copy — like the one in this repo — for
anything public.

## Setup

1. **Import the workflow.** In n8n: `Workflows → Import from File` and select
   `Knatware_LLM_Notebook_Generator_n8n_workflow.json`.
2. **Place the notebook template.** The `Fill Notebook Template` code node has the
   entire notebook (with `{{PLACEHOLDER}}` tokens) embedded directly in its code —
   you do **not** need to upload `Knatware_LLM_FineTuning_V3_Colab_Template.ipynb`
   anywhere. It's included in this repo purely as a human-readable reference for
   what the generated notebook looks like before substitution.
3. **Connect Google Drive** (only if you'll use the Colab-push option): open the
   **Upload to Google Drive** node and connect a Google Drive OAuth2 credential.
   This has to be authorized inside your own n8n instance — it can't be
   pre-filled by importing the JSON.
4. **Set your Hugging Face token** (optional) — open the **Config (edit token
   here)** node and edit the `HF_TOKEN` constant. See
   [Configuration: your Hugging Face token](#configuration-your-hugging-face-token)
   for details; this is opt-in and the workflow works fine without it.
5. **Activate the workflow**, then open the **Step 1** form's production URL
   (found on the `Step 1 – Search Base Models` trigger node) to run it.

---

## Using it

1. **Step 1:** optionally enter a search keyword (e.g. `qwen`, `llama`, `mistral`),
   a size preference, and your Hugging Face username. (No token field here —
   see [Configuration: your Hugging Face token](#configuration-your-hugging-face-token).)
2. **Step 2:** pick a model from the live dropdown, or type any Hugging Face model
   id in the override field.
3. **Step 3:** pick a training dataset from the live dropdown (ranked by relevance
   to your model), or type any dataset id.
4. **Step 4:** every hyperparameter field shows a recommended value as its
   placeholder — leave blank to accept it, or type your own. Choose **Download**
   or **Upload to Google Drive & open in Colab** at the bottom.
5. You're done — either the `.ipynb` downloads immediately, or you get a direct
   Colab link.

---

## Customizing

- **Change the dataset ranking logic:** edit the `Build Dataset Search Params` and
  `Build Dataset Dropdown` code nodes — currently ranks by task + a "family"
  keyword parsed from the model id.
- **Change hyperparameter defaults:** edit the `LORA_TARGETS_BY_TYPE` map and the
  various `recommended*` calculations in `Derive Recommendations`.
- **Change the notebook itself:** edit
  `Knatware_LLM_FineTuning_V3_Colab_Template.ipynb`, then regenerate the embedded
  `TEMPLATE` string in the `Fill Notebook Template` node (see
  [Regenerating the embedded template](#regenerating-the-embedded-template) below).
  Editing the JSON's embedded copy directly is not recommended — it's a single
  ~43,000-character escaped string.

### Regenerating the embedded template

If you edit the reference `.ipynb` file, regenerate the `TEMPLATE` constant with
Node.js so the escaping is guaranteed correct:

```js
const fs = require("fs");
const nb = JSON.parse(fs.readFileSync("Knatware_LLM_FineTuning_V3_Colab_Template.ipynb", "utf-8"));
const templateLine = "const TEMPLATE = " + JSON.stringify(JSON.stringify(nb)) + ";";
fs.writeFileSync("template_line.js", templateLine);
```

Then paste the contents of `template_line.js` into the `Fill Notebook Template`
code node, replacing the existing `const TEMPLATE = "...";` line (everything else
in that node — the replacement logic, `pyString`/`pyNumber`/`pyBool`/`pyList`
helpers, and the final packaging — stays the same).

Every placeholder token in the notebook must exactly match a key in the
`replacements` object inside `Fill Notebook Template` (case-sensitive, double
curly braces: `{{MODEL_NAME}}`). A leftover-placeholder check runs automatically
and throws a clear error if anything is missed.

---

## Notes & known limitations

- **Dataset relevance is heuristic**, not authoritative — Hugging Face doesn't
  expose "datasets used to train model X" via a public API, so this workflow
  ranks by the model's declared task plus a family keyword parsed from its repo
  id. Always sanity-check the chosen dataset.
- **Self-healing dropdowns:** if a dropdown selection somehow fails to reach the
  resolve step (blank submission), the workflow falls back to the first option it
  offered rather than crashing, and shows an orange warning banner on Step 4 so
  you know to double-check `MODEL_NAME`/`DATASET_NAME` before running the
  notebook.
- **Security:** if you fill in the `HF_TOKEN` constant (see
  [Configuration: your Hugging Face token](#configuration-your-hugging-face-token)),
  it's written in plaintext into every generated notebook's login cell
  (`HF_TOKEN = "..."`) so it can log in automatically — that's unavoidable if
  you want the notebook to be immediately runnable. Treat generated notebooks
  as sensitive: don't commit them to a public repo. And remember the `Config`
  node itself holds that same value in plain text if you fill it in — don't
  commit a filled-in copy of the *workflow* either.
- **n8n's paired-item lookup (`.item`) is unreliable across Merge nodes.** Every
  code node in this workflow uses `$('Node Name').first()` instead of
  `$('Node Name').item` for exactly this reason — keep that convention if you add
  more nodes.

---

## License / attribution

Notebook and workflow: © Knatware Technology — developed by Kayode Okosi, LLM
Developer.
