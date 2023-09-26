#!/bin/bash

# Build and push vLLM docker image to AWS ECR.

set -eo pipefail


IMAGE="692474966980.dkr.ecr.us-west-2.amazonaws.com/fractal-chat-service"

# Build and push image for fastapi server
TAG="vllm-$(git rev-parse HEAD)"

DOCKER_BUILDKIT=1 docker build -f launch_deployment/Dockerfile -t "$IMAGE:$TAG" .

echo Pushing "$IMAGE:$TAG"
docker push "$IMAGE:$TAG"