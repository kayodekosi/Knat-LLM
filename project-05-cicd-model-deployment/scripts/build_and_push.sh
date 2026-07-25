#!/usr/bin/env bash
# Convenience script mirroring the Jenkinsfile's build+push stages, useful for
# local testing of the image before it ever touches the CI pipeline.
set -euo pipefail

REGISTRY="${REGISTRY:-registry.knatware.com/knat-llm}"
IMAGE_NAME="${IMAGE_NAME:-llm-server}"
TAG="${1:-local-dev}"

echo "Building ${REGISTRY}/${IMAGE_NAME}:${TAG}..."
docker build -t "${REGISTRY}/${IMAGE_NAME}:${TAG}" .

echo "Running smoke test..."
docker run -d --name smoke-test -p 8010:8000 "${REGISTRY}/${IMAGE_NAME}:${TAG}"
sleep 15
curl --fail http://localhost:8010/health
docker rm -f smoke-test

echo "Pushing ${REGISTRY}/${IMAGE_NAME}:${TAG}..."
docker push "${REGISTRY}/${IMAGE_NAME}:${TAG}"
