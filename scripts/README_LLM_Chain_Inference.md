# Knatware LLM Chain — Open-Source Model Inference (OpenRouter) — n8n Workflow

A minimal, 5-node chat workflow that sends messages to a **free, open-weight
model hosted on OpenRouter** — fully cloud-hosted, nothing to install, no
credit card required. Meant as a lightweight companion to the
[Notebook Generator workflow](README_Notebook_Generator.md) — fine-tune a model
with that workflow, push it somewhere it can be served, and talk to it here.

**File:** `Knatware_LLM_Chain_Inference_n8n_workflow.json`

---

## What it does

1. n8n's built-in **Chat** panel captures whatever you type.
2. A **Basic LLM Chain** node forwards it to a connected language model, with a
   short system prompt.
3. An **OpenRouter Chat Model** node runs an open-weight model in the cloud via
   [OpenRouter](https://openrouter.ai) and returns the reply.

No GPU, no disk space, no Docker, no local model server — everything runs on
OpenRouter's infrastructure, for free on their `:free` tier.

### Why OpenRouter, and not Ollama or a repointed OpenAI node?

Two earlier versions of this workflow were tried and dropped:

- **Ollama** (self-hosted, local) works well but needs several GB of disk space
  per model plus a machine to run it on — ruled out when that's not available.
- **Hugging Face via a repointed OpenAI Chat Model node** (setting its Base URL
  to Hugging Face's router) looked correct on paper, but in testing the Base
  URL override didn't reliably take effect at request time — requests silently
  went to OpenAI's real API instead, surfacing as an **OpenAI billing/rate-limit
  error** with no Hugging Face involved at all.

OpenRouter avoids both problems: it's fully cloud-hosted (no install, no disk
space), and n8n ships a **dedicated** `OpenRouter Chat Model` node with its own
credential type — not a generic node repointed via a URL override — so there's
no ambiguity about where requests actually go.

---

## Architecture

```
When chat message received  ──(main)──▶  Basic LLM Chain  ◀──(ai_languageModel)──  OpenRouter Chat Model
   (Chat Trigger)                       (root/chain node)                          (open-weight model, cloud-hosted, free tier)
```

| Node                             | Type                                       | Role                                                   |
|-----------------------------------|-----------------------------------------------|---------------------------------------------------------|
| When chat message received        | `@n8n/n8n-nodes-langchain.chatTrigger`         | Opens n8n's built-in chat UI as the entry point         |
| Basic LLM Chain                   | `@n8n/n8n-nodes-langchain.chainLlm`            | Takes `chatInput` automatically, applies a system prompt, calls the model |
| OpenRouter Chat Model             | `@n8n/n8n-nodes-langchain.lmChatOpenRouter`    | The actual open-weight model, served by OpenRouter      |
| Overview / OpenRouter Setup       | `n8n-nodes-base.stickyNote`                    | In-canvas documentation                                 |

---

## Prerequisites

- A running n8n instance with the `@n8n/n8n-nodes-langchain` package available
  (bundled by default in current n8n releases).
- A free [OpenRouter](https://openrouter.ai) account and API key — no credit
  card needed to use `:free` models.

---

## Setup

1. **Get an API key:** [openrouter.ai/keys](https://openrouter.ai/keys) — sign
   up, create a key.
2. **Import the workflow.** In n8n: `Workflows → Import from File` and select
   `Knatware_LLM_Chain_Inference_n8n_workflow.json`.
3. **Connect a credential.** Open the **OpenRouter Chat Model** node →
   **Credential to connect with** → **Create new credential** → **OpenRouter
   API** → paste your key.
4. **Leave the Model field as-is to start** (`openrouter/free` — see
   [How the default model works](#how-the-default-model-works) below), or pick a
   specific model (see [Changing the model](#changing-the-model)).
5. **Activate the workflow.**
6. Click **Chat** at the bottom of the n8n canvas and start talking to it.

---

## How the default model works

The Model field ships set to **`openrouter/free`** — OpenRouter's own **Free
Models Router**, not a specific model. It automatically picks a free,
open-weight model for each request (filtering for whatever capabilities the
request needs), and costs nothing.

This is deliberate: OpenRouter's specific free-model roster **changes
regularly** — models get added and delisted week to week as providers rotate
promotional free tiers (for example, the free tiers for Llama 3.3 70B and
several Qwen models were both pulled just days before this note was written).
Hardcoding one specific `model:free` slug as the default risks it silently
being gone by the time you actually use this workflow — exactly the failure
this workflow is meant to avoid. `openrouter/free` sidesteps that by always
routing to *something* currently free.

---

## Changing the model

1. Open the **OpenRouter Chat Model** node.
2. Click the **Model** field and pick any specific model instead of the
   `openrouter/free` router — e.g.:
   - `meta-llama/llama-4-maverick:free`
   - `qwen/qwen3-coder:free`
   - `google/gemma-4-31b-it:free`
   - `deepseek/deepseek-v4-flash:free` (check it's still free — see note above)
3. **Check the live free-model list before committing to one:**
   [openrouter.ai/models?max_price=0](https://openrouter.ai/models?max_price=0)
   — any model id ending in `:free` costs nothing; anything else bills your
   OpenRouter balance per token.
4. Save, and the very next chat message uses the new model.

**Using a paid model instead:** drop the `:free` suffix (e.g.
`meta-llama/llama-3.3-70b-instruct`) and add credit to your OpenRouter account
— useful if the free tier's rate limits are too restrictive for what you're
doing.

---

## Customizing further

- **System prompt:** edit the `messages.messageValues` array in the **Basic LLM
  Chain** node.
- **Generation parameters** (temperature, max tokens, etc.): open the
  **OpenRouter Chat Model** node's **Options** panel in the n8n UI and set them
  there.
- **Swap the trigger:** replace **When chat message received** with a Webhook,
  Slack, or Telegram trigger node (map its incoming text field to `chatInput`
  going into **Basic LLM Chain**, or add a Set node in between) to expose this as
  an API or chatbot instead of n8n's built-in chat panel.
- **Add memory:** connect a `Simple Memory` node
  (`@n8n/n8n-nodes-langchain.memoryBufferWindow`) to **Basic LLM Chain**'s memory
  input for multi-turn conversation history.

---

## Notes & known limitations

- The **Basic LLM Chain** node expects an incoming `chatInput` field when its
  Prompt is set to "Take from previous node automatically" (which this workflow
  uses). If you swap the trigger for something else, make sure the field feeding
  it is named `chatInput`, or add an **Edit Fields (Set)** node to rename it.
- **Free-tier rate limits** (subject to change on OpenRouter's side): roughly 20
  requests/minute and 50–1000 requests/day depending on account history —
  plenty for personal use and testing, not for production traffic. If you hit
  limits often, either add a small balance to raise the daily cap, or switch to
  a paid (non-`:free`) model.
- **Free models can have higher latency and less consistent availability** than
  paid ones, especially during peak hours — this is expected behavior of
  promotional free tiers, not a bug in this workflow.
- If you ever see an error like `invalid model ID` or "not supported for
  task ...", that specific model's free-tier availability has likely changed —
  reopen the Model field and pick a currently-listed one, or fall back to
  `openrouter/free`.

---

## License / attribution

© Knatware Technology — developed by Kayode Okosi, LLM Developer.
