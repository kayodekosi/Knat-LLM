# Knat LLM

Enterprise-grade AI/ML/LLM infrastructure, built and published in the open by
**Knatware Technology**.

Knat LLM is a series of seven hands-on reference projects that together cover the full lifecycle
of running large language models in production: from containerized data services, through
self-hosted model serving and Kubernetes orchestration, to retrieval-augmented applications,
CI/CD for model updates, optimization studies, and multi-GPU scaling. Each project is a complete,
runnable repository with its own README, architecture notes, and setup instructions — designed to
be read and used independently, while building on the ones before it.

## The projects

| # | Project | What it covers |
|---|---------|-----------------|
| 1 | [Dockerized FastAPI + Postgres](https://github.com/Knat-LLM/project-01-dockerized-fastapi-postgres) | A containerized CRUD service — SQL fluency and container basics |
| 2 | [Local LLM Server](https://github.com/Knat-LLM/project-02-local-llm-server) | Self-hosted Llama 3 / Mistral via vLLM, OpenAI-compatible API, throughput benchmarking |
| 3 | [Kubernetes Deployment](https://github.com/Knat-LLM/project-03-k8s-llm-deployment) | The LLM server on k3s/minikube, with HPA autoscaling and a Grafana dashboard |
| 4 | [RAG Pipeline](https://github.com/Knat-LLM/project-04-rag-pipeline) | PDF ingestion → embeddings → Elasticsearch → Haystack → chat UI, with tracing |
| 5 | [CI/CD for Models](https://github.com/Knat-LLM/project-05-cicd-model-deployment) | Jenkins + ArgoCD, canary rollouts gated on real latency/error metrics |
| 6 | [Optimization Study](https://github.com/Knat-LLM/project-06-quantization-study) | FP16 vs INT8 vs AWQ — measured latency, throughput, and quality tradeoffs |
| 7 | [Multi-GPU Serving](https://github.com/Knat-LLM/project-07-multi-gpu-serving) | Tensor-parallel deployment across 2 rented GPUs (RunPod/Lambda/Vast.ai) |

## How the series fits together

```
Project 1 ──▶ Project 2 ──▶ Project 3 ──▶ Project 5
(data layer)  (model serving) (orchestration) (delivery pipeline)
                    │
                    ├──▶ Project 4 (application layer: RAG)
                    ├──▶ Project 6 (optimization: quantization)
                    └──▶ Project 7 (scaling: multi-GPU)
```

Projects 1–3 build the infrastructure backbone. Project 4 turns that infrastructure into an
actual user-facing application. Project 5 makes model updates safe to ship continuously. Projects
6 and 7 answer the two questions every AI platform team eventually faces: *how do we make this
model cheaper/faster to run*, and *how do we serve a model too large for one GPU*.

## About Knatware Technology

Knatware Technology builds and operates AI/ML/LLM infrastructure for organizations running
models at production scale — from inference platform design and GPU capacity planning to
retrieval-augmented application delivery and MLOps/LLMOps pipelines. Knat LLM is our open
reference series, published to demonstrate the same patterns we implement for clients.

## Enquiries & implementation support

Each project repository includes setup instructions, architecture notes, and natural next steps
for extending it. For enquiries, custom implementation, or adapting any of these projects to a
production environment, contact **kayode@knatware.com**.

---
© Knatware Technology
