# Project 5 — CI/CD for Models: Jenkins + ArgoCD Canary Rollout

**Org:** Knat LLM · **Maintained by:** Knatware Technology

A GitOps-style delivery pipeline for the model server: **Jenkins** builds, tests, and pushes a new
image on every push, then hands off to **ArgoCD**, which rolls the new version out gradually as a
**canary release** — watching real latency and error-rate metrics before shifting more traffic to
it. This is the "how does a new model version reach production safely" project in the series.

## Why canary, not a straight rollout

Model updates are riskier than typical service deploys — a new checkpoint, quantization, or
prompt template change can silently degrade answer quality or blow up latency in ways a basic
health check won't catch. A canary rollout, gated on real metrics (not just "is the pod running"),
is the honest way to deploy model changes to an enterprise platform.

## Architecture

```
 git push (main) ─▶ Jenkins pipeline
                        │  1. build image (tagged with git SHA)
                        │  2. smoke test (container starts, /health passes)
                        │  3. push image to registry
                        │  4. bump image tag in the GitOps repo
                        ▼
                  GitOps repo (manifest change)
                        │
                        ▼
              ArgoCD (auto-sync, watching GitOps repo)
                        │
                        ▼
        Argo Rollouts canary: 10% → 25% → analysis gate → 50% → 100%
                        │
                        ▼
         AnalysisTemplate checks P95 latency + error rate at each step
         (queries Prometheus — the same stack wired up in Project 3)
```

## Components

| File                              | Purpose                                                            |
|------------------------------------|----------------------------------------------------------------------|
| `Jenkinsfile`                      | Build → smoke test → push → update GitOps manifest                    |
| `argocd/application.yaml`          | ArgoCD `Application` pointing at the GitOps repo, auto-sync enabled  |
| `k8s/rollout.yaml`                 | Argo Rollouts `Rollout` resource replacing a plain Deployment          |
| `k8s/analysis-template.yaml`       | Automated pass/fail gate: P95 latency < 2.5s and error rate < 2%      |
| `scripts/build_and_push.sh`        | Local script to build/test/push an image outside of Jenkins           |

## Prerequisites

- A Jenkins instance with Docker available to its agents, and credentials configured for:
  `registry-creds` (container registry) and `gitops-deploy-key` (SSH key with push access to the
  GitOps repo)
- ArgoCD installed on the target cluster, with `argocd-application-controller` able to reach the
  GitOps repo
- **Argo Rollouts** installed (`kubectl argo rollouts` plugin, plus its controller) for the canary
  mechanics in `rollout.yaml`
- The Prometheus stack from **Project 3** already scraping `llm-server` metrics, since the
  `AnalysisTemplate` depends on those queries existing

## How a deploy actually flows

1. A developer pushes to `main`. Jenkins picks it up via the configured webhook/poll.
2. Jenkins builds the image, tags it with the short git SHA, and runs a smoke test — a real
   container boot plus a `/health` check, not just a build success.
3. On success, Jenkins pushes the image and edits the GitOps repo's Helm values (or Kustomize
   overlay) to point at the new tag, then commits and pushes that change.
4. ArgoCD detects the GitOps repo change (it's watching, not being pushed to directly) and syncs
   the cluster to match — this is what makes it GitOps: **the GitOps repo is the source of truth**,
   not a `kubectl apply` run from CI.
5. Because the workload is an Argo Rollouts `Rollout` (not a plain `Deployment`), the sync doesn't
   immediately replace all pods. It steps through the canary weights in `rollout.yaml`, and at the
   defined `analysis` step, automatically checks the metrics query in `analysis-template.yaml`. A
   failed check pauses (or can be configured to auto-abort) the rollout — the new version never
   sees full traffic.

## Watching a rollout

```bash
kubectl argo rollouts get rollout llm-server -n knat-llm --watch
```

This shows live traffic-weight percentages and the current step, including whether the analysis
gate passed.

## What this project deliberately practices

- The distinction between CI (build/test/push) and CD (GitOps sync) as separate concerns
- Why the GitOps repo — not the CI pipeline — should be the thing that actually changes cluster state
- Progressive delivery (canary) driven by real application metrics, not just pod readiness
- Designing an automated rollback gate before a bad model version reaches all users

## Natural next steps

- Add automated rollback (`abort` on analysis failure) instead of only pausing
- Add a pre-canary offline eval step (accuracy/quality regression check) before any traffic shifts
- Extend the analysis template with a business metric (e.g. RAG groundedness score) alongside latency/error rate

## Enquiries & implementation support

For enquiries or help wiring this into an existing Jenkins/ArgoCD estate, contact
**kayode@knatware.com**.

---
© Knatware Technology — part of the **Knat LLM** project series.
