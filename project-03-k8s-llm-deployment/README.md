# Project 3 — Kubernetes Deployment of the LLM Server (HPA + Grafana)

**Org:** Knat LLM · **Maintained by:** Knatware Technology

Takes the vLLM server from **Project 2** off a single Docker Compose host and onto Kubernetes —
tested against a local **k3s** or **minikube** cluster — with a **HorizontalPodAutoscaler** and a
**Grafana dashboard** for real observability. This is where the series stops being "a container
that runs on my machine" and starts being "a workload a platform team could actually operate."

## Why this project exists

Enterprise AI platforms don't run inference servers as standalone containers — they run them as
Kubernetes workloads so they get restart policies, rolling updates, resource quotas, and
autoscaling for free. This project deliberately practices that operational layer using the same
model server built in Project 2, so the only new variable is the orchestration, not the model.

## Architecture

```
                    ┌───────────────────────────────────────┐
                    │              knat-llm namespace         │
                    │                                          │
   Client ───────▶  │   Service (ClusterIP)                    │
                    │        │                                 │
                    │        ▼                                 │
                    │   Deployment: llm-server (vLLM pod)       │
                    │        ▲                                 │
                    │        │ scales                          │
                    │   HorizontalPodAutoscaler (CPU, 1→4 pods) │
                    │        │                                 │
                    │   PersistentVolumeClaim (HF model cache)  │
                    └───────────────────────────────────────┘
                                   │
                     ServiceMonitor → Prometheus → Grafana
```

## Prerequisites

- A local cluster: **k3s** (`curl -sfL https://get.k3s.io | sh -`) or **minikube**
  (`minikube start --driver=docker`)
- `kubectl` configured against that cluster
- NVIDIA GPU Operator installed on the cluster if you want real GPU scheduling (otherwise remove
  the `nvidia.com/gpu` resource requests/limits to run on CPU for structural testing only)
- `kube-prometheus-stack` (Prometheus Operator + Grafana) installed via Helm for the monitoring layer

## Deploying

```bash
git clone <this-repo-url>
cd project-03-k8s-llm-deployment

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Optional, requires the Prometheus Operator CRDs already installed:
kubectl apply -f k8s/servicemonitor.yaml
```

Check status:

```bash
kubectl -n knat-llm get pods,svc,hpa
```

Port-forward to test locally:

```bash
kubectl -n knat-llm port-forward svc/llm-server-svc 8000:8000
curl http://localhost:8000/health
```

## Autoscaling behavior

`k8s/hpa.yaml` scales the deployment between 1 and 4 replicas based on CPU utilization (target
70%). This is intentionally the simplest possible signal to get autoscaling working end-to-end
first. The README on the manifest itself notes the honest limitation: **CPU is a weak proxy for
LLM-serving load** — a GPU-bound inference server can be CPU-idle while its GPU queue is
saturated. The natural production upgrade is scaling on a custom metric (e.g. request queue depth
or GPU utilization) surfaced through the **Prometheus Adapter**, referenced as a next step in the
manifest comments.

## Grafana dashboard

`grafana/dashboard.json` defines four panels: requests/sec, P95 latency, GPU utilization, and
active replica count. Import it via **Grafana → Dashboards → Import**, pointed at a Prometheus
data source that's scraping both the `ServiceMonitor` here and a DCGM exporter (for GPU metrics)
and `kube-state-metrics` (for replica counts).

## Load-testing the HPA

To watch autoscaling happen live, generate sustained load against the port-forwarded endpoint
(e.g. with `hey`, `locust`, or the benchmark script from Project 2 run with a high `--concurrency`)
and watch:

```bash
kubectl -n knat-llm get hpa -w
```

## What this project deliberately practices

- Translating a Docker Compose service into proper Kubernetes manifests
- Resource requests/limits and GPU scheduling in Kubernetes
- Readiness/liveness probes tuned for a slow-starting (model-loading) container
- HPA mechanics, and — just as importantly — the honest limits of CPU-based autoscaling for GPU workloads
- Wiring a workload into a Prometheus/Grafana observability stack

## Enquiries & implementation support

For enquiries, custom implementation, or migrating this to OpenShift or a managed cluster, contact
**kayode@knatware.com**.

---
© Knatware Technology — part of the **Knat LLM** project series.
