# Knatware LLM Chain — Open-Source Model Inference (Hugging Face) — n8n Workflow

A minimal, 5-node chat workflow that sends messages to an **open-source language
model hosted on Hugging Face**, via n8n's built-in chat interface. Meant as a
lightweight companion to the [Notebook Generator workflow](README_Notebook_Generator.md) —
fine-tune a model with that workflow, then swap its repo id into this one to talk
to it.

**File:** `Knatware_LLM_Chain_Inference_n8n_workflow.json`

---

## What it does

1. n8n's built-in **Chat** panel captures whatever you type.
2. A **Basic LLM Chain** node forwards it to a connected language model, with a
   short system prompt.
3. A **Hugging Face Inference Model** node runs an open-source model on Hugging
   Face's Inference API servers and returns the reply.

No local GPU, no Docker, no self-hosted model server required — inference happens
on Hugging Face's infrastructure.

---

## Architecture

```
When chat message received  ──(main)──▶  Basic LLM Chain  ◀──(ai_languageModel)──  Hugging Face Inference Model
   (Chat Trigger)                       (root/chain node)                          (open-source model, swappable)
```

| Node                             | Type                                                  | Role                                                   |
|-----------------------------------|--------------------------------------------------------|---------------------------------------------------------|
| When chat message received        | `@n8n/n8n-nodes-langchain.chatTrigger`                 | Opens n8n's built-in chat UI as the entry point         |
| Basic LLM Chain                   | `@n8n/n8n-nodes-langchain.chainLlm`                    | Takes `chatInput` automatically, applies a system prompt, calls the model |
| Hugging Face Inference Model       | `@n8n/n8n-nodes-langchain.lmOpenHuggingFaceInference`  | The actual open-source model, called via Hugging Face's free Inference API |
| Overview / Hugging Face Setup     | `n8n-nodes-base.stickyNote`                            | In-canvas documentation                                 |

---

## Prerequisites

- A running n8n instance with the `@n8n/n8n-nodes-langchain` package available
  (bundled by default in current n8n releases).
- A free Hugging Face account and access token:
  [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — Read
  access is enough for public models.

---

## Setup

1. **Import the workflow.** In n8n: `Workflows → Import from File` and select
   `Knatware_LLM_Chain_Inference_n8n_workflow.json`.
2. **Connect a Hugging Face credential.** Open the **Hugging Face Inference Model**
   node → **Credential to connect with** → **Create new credential** → paste your
   token.
3. **Check the Model field** on that same node (default:
   `Qwen/Qwen2.5-1.5B-Instruct`) — change it to whatever model you want to talk to
   (see [Changing the model](#changing-the-model) below).
4. **Activate the workflow.**
5. Click **Chat** at the bottom of the n8n canvas and start talking to it.

---

## Changing the model

This is a single text field — no redeploy needed:

1. Open the **Hugging Face Inference Model** node.
2. Edit **Model** to any Hugging Face repo id you have inference access to, e.g.:
   - `Qwen/Qwen2.5-1.5B-Instruct` (default — small, fast, good for testing)
   - `mistralai/Mistral-7B-Instruct-v0.3`
   - `meta-llama/Llama-3.2-3B-Instruct` (gated — see below)
   - `HuggingFaceH4/zephyr-7b-beta`
   - **The model you just fine-tuned** with the
     [Notebook Generator workflow](README_Notebook_Generator.md) and pushed to
     the Hub (i.e. its `HUB_REPO_ID` or `DEPLOY_MODEL_REPO_ID` from that run).
3. Save, and the very next chat message uses the new model — no other node needs
   to change.

**Gated models** (Llama, Gemma, etc.) require visiting the model's page on
huggingface.co, accepting its license, and using a token with access to it.

**Model availability:** Hugging Face's free Inference API only serves a subset of
models at any time, and can return a "model is loading" response on first call for
models that haven't been used recently — retry after a few seconds if that
happens. Smaller instruct models (1B–8B parameters) are the most reliably
available on the free tier.

---

## Customizing further

- **System prompt:** edit the `messages.messageValues` array in the **Basic LLM
  Chain** node.
- **Generation parameters** (temperature, max tokens, etc.): open the **Hugging
  Face Inference Model** node's **Options** panel in the n8n UI and set them
  there directly — this workflow ships with defaults only, since the exact
  option keys can shift between n8n versions.
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
- **Sub-node expression quirk:** the Hugging Face Inference Model node is a
  "sub-node" in n8n's LangChain architecture — any expression inside it always
  resolves against the *first* input item, not each item in turn (this only
  matters if you start batching multiple chat inputs through the workflow at
  once).
- This workflow calls Hugging Face's **free, shared Inference API** — for
  production traffic or guaranteed uptime/latency, consider a paid **Hugging Face
  Inference Endpoint** instead (same node, point the credential/model at your
  dedicated endpoint).

---

## License / attribution

© Knatware Technology — developed by Kayode Okosi, LLM Developer.
