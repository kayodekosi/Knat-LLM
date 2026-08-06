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
Step 1 (search) ──▶ Search HF Models ──▶ Build Model Dropdown
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
                                       │
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

23 nodes total, including two sticky notes with in-canvas setup instructions.

---

## Prerequisites

- A running n8n instance (self-hosted or cloud) with the **`@n8n/n8n-nodes-langchain`**
  community/core nodes available for the sub-workflow described in the companion
  [LLM Chain Inference workflow](README_LLM_Chain_Inference.md) — not required for
  *this* workflow, which only uses core n8n nodes (Form, HTTP Request, Code, IF,
  Google Drive, Merge).
- A Google account, **if** you plan to use the "push to Google Drive & open in
  Colab" delivery option.
- (Optional, admin-side) A Hugging Face account and access token, attached as
  an n8n **Header Auth credential** on the 5 Hugging Face HTTP nodes — needed
  for gated models (Llama, Gemma, etc.) or to raise Hugging Face API rate
  limits. See [Secret / constant parameters](#secret--constant-parameters).
  Not something end users of the form need to provide.

---

## Secret / constant parameters

This workflow deliberately separates **per-run choices** (collected via the
form: model, dataset, hyperparameters, delivery method) from **secret,
constant values** that should never be typed into a form or vary per user.

There are two independent secret-parameter concerns here, handled two
different ways — because they have genuinely different constraints:

### 1. Authenticating this workflow's own Hugging Face API calls

The 5 HTTP Request nodes that call huggingface.co (model search, model info,
dataset search ×2, dataset info) ship with **Authentication: None** — this
works fine for public models/datasets, just subject to Hugging Face's lower
unauthenticated rate limit.

To authenticate them, **on each of the 5 nodes**: set **Authentication** to
**Generic Credential Type** → **Generic Auth Type**: **Header Auth** → create
a credential with **Name** = `Authorization` and **Value** =
`Bearer <your Hugging Face token>`.

This uses n8n's built-in **Credentials** store — encrypted, configured once,
never appears in the exported workflow JSON, and never exposed to whoever
fills out the form. We use Credentials here rather than an environment
variable (`$env`) because many self-hosted n8n instances set
`N8N_BLOCK_ENV_ACCESS_IN_NODE`, which denies workflows *any* `$env` access —
a real, common security policy on self-hosted instances, not a bug in this
workflow. Credentials aren't affected by that setting.

### 2. Injecting a token into the *generated notebook's* login cell

This is a different problem: the generated `.ipynb` needs an actual token
string written into its text so it can auto-login when someone opens it in
Colab. n8n Credentials can't help here — by design, a workflow can *use* a
credential to authenticate a request, but can never *read back* the raw
secret value as data. And since `$env` may be blocked (see above), that
route isn't reliably available either.

So this value lives as a single, clearly-labeled constant —
**`HF_TOKEN_CONSTANT`** — at the very top of the **Fill Notebook Template**
code node. It ships **blank by default**, on purpose: this workflow is meant
to be published in a public GitHub repo, and a filled-in token there would be
committed in plain text along with everything else in the file.

If you want every generated notebook to auto-login with a real token, the
comment block around `HF_TOKEN_CONSTANT` documents two options:

- **If your n8n instance allows `$env` access:** set `KNATWARE_HF_TOKEN` as an
  environment variable on the instance, then change the constant to read
  `$env.KNATWARE_HF_TOKEN || ""` instead of `""`.
- **If not (or you'd rather not touch `$env` at all):** type your token
  directly into the constant. This works regardless of
  `N8N_BLOCK_ENV_ACCESS_IN_NODE`, but means the token now lives in plain text
  inside this workflow file — **do not commit a filled-in copy to a public
  repo** if you do this. Keep a private copy with the real value, and a
  separate blank copy (like the one in this repo) for anything public.

Either way, if `HF_TOKEN_CONSTANT` stays blank, the generated notebook's login
cell prints an explanatory message instead of failing, and whoever runs the
notebook can paste their own token in manually — see the comment block above
`HF_TOKEN = "{{HF_TOKEN}}"` in
`Knatware_LLM_FineTuning_V3_Colab_Template.ipynb`, Section 2, for the exact
behavior and reasoning (it's documented there too, not just here).

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
4. **Authenticate Hugging Face calls** (optional) and **set a notebook login
   token** (optional) — see
   [Secret / constant parameters](#secret--constant-parameters) for exact
   steps; both are opt-in and the workflow works without either.
5. **Activate the workflow**, then open the **Step 1** form's production URL
   (found on the `Step 1 – Search Base Models` trigger node) to run it.

---

## Using it

1. **Step 1:** optionally enter a search keyword (e.g. `qwen`, `llama`, `mistral`),
   a size preference, and your Hugging Face username. (No token field here —
   see [Secret / constant parameters](#secret--constant-parameters).)
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
- **Security:** if you fill in `HF_TOKEN_CONSTANT` (see
  [Secret / constant parameters](#secret--constant-parameters)), it's written
  in plaintext into every generated notebook's login cell (`HF_TOKEN = "..."`)
  so it can log in automatically — that's unavoidable if you want the notebook
  to be immediately runnable. Treat generated notebooks as sensitive: don't
  commit them to a public repo. And remember `HF_TOKEN_CONSTANT` itself is
  plain text inside the workflow file if you fill it in directly rather than
  via `$env` — don't commit a filled-in copy of the *workflow* either.
- **n8n's paired-item lookup (`.item`) is unreliable across Merge nodes.** Every
  code node in this workflow uses `$('Node Name').first()` instead of
  `$('Node Name').item` for exactly this reason — keep that convention if you add
  more nodes.

---

## License / attribution

Notebook and workflow: © Knatware Technology — developed by Kayode Okosi, LLM
Developer.
