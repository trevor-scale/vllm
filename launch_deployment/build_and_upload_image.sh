#!/bin/bash

# Build and push vLLM docker image to AWS ECR.

set -eo pipefail

IMAGE_TAG="vllm-$(git rev-parse HEAD)"
ACCOUNT=692474966980

# Todo stand up ECR repo for this project instead of being lazy
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.us-west-2.amazonaws.com
DOCKER_BUILDKIT=1 docker build -f launch_deployment/Dockerfile -t $ACCOUNT.dkr.ecr.us-west-2.amazonaws.com/vllm:$IMAGE_TAG .
docker push $ACCOUNT.dkr.ecr.us-west-2.amazonaws.com/fractal-chat-service:$IMAGE_TAG