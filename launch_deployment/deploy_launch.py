"""Contains code for working with launch endpoint (Checking endpoint, testing, deploying, etc)"""
# python launch_deployment/deploy_launch.py --image_tag vllm-63aeeb78b835232d7f0c7bcbfe1f7d364b83bb06 --redeploy
import argparse
import os
from time import sleep
from typing import Dict

import launch
from launch_internal import get_launch_client
from pydantic import BaseModel

LAUNCH_ENV = os.getenv("LAUNCH_ENV", "prod")  # or "training"

BUNDLE_TAG = "vllm-guided-decoding"
ENDPOINT_TYPE = "async"
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", f"{BUNDLE_TAG}-{ENDPOINT_TYPE}-service")
FEDERAL_ID = "63bf16fbc65a8abb84ad8619"

launch_client = get_launch_client(api_key=FEDERAL_ID, env=LAUNCH_ENV, v1=True)

test_regexes = [
    r"\s*([Yy]es|[Nn]o|[Nn]ever|[Aa]lways)",
    '{\n  "gender": ("m"|"f"),\n  "height_inches": (0|[1-9][0-9]*),\n  "weight_lbs": (0|[1-9][0-9]*)\n}',
]
test_prompts = [
    "Is 1+1=2? ",
    "Write a json blob describing a tennis player including the fields gender (m/f), height (inches) and weight (lbs): ",
]


def get_bundle_config(bundle_name: str, image_tag: str) -> Dict:
    return {
        "model_bundle_name": bundle_name,
        "request_schema": BaseModel,
        "response_schema": BaseModel,
        "repository": "fractal-chat-service",
        "tag": image_tag,
        "command": [
            "dumb-init",
            "--",
            "python",
            "-m",
            "vllm_server" "--model",
            "meta-llama/Llama-2-7b-chat-hf",
            "--tensor-parallel-size",
            "2",
            "--port",
            "5005",
            "--max-num-batched-tokens",
            "4096",
        ],
        "readiness_initial_delay_seconds": 20,
        "env": {"AWS_PROFILE": "ml-worker", "DERIVE_CLASSIFICATIONS": "0"},
    }


def get_pytorch_service_params(endpoint_name, endpoint_type):
    return {
        "endpoint_name": endpoint_name,
        "cpus": 8,
        "memory": "40Gi",
        "gpus": 2,
        "gpu_type": "nvidia-ampere-a10",
        "min_workers": 1,
        "max_workers": 2,
        "per_worker": 10,
        "endpoint_type": endpoint_type,
        "labels": {"team": "federal", "product": "fractal"},
    }


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--endpoint_test", action="store_true")
    argparser.add_argument("--redeploy", action="store_true")
    argparser.add_argument("--update_bundle", action="store_true")
    argparser.add_argument("--image_tag", default="")
    argparser.add_argument("--check_endpoint", action="store_true")

    args = argparser.parse_args()
    if args.check_endpoint:
        for endpoint in launch_client.list_model_endpoints():
            print(endpoint.model_endpoint)

    if args.endpoint_test:
        endpoint = launch_client.get_model_endpoint(ENDPOINT_NAME)

        for prompt, regex in zip(test_prompts, test_regexes):
            print(f"Request: {prompt}")

            request_body = {
                "prompt": prompt,
                "temperature": 0.0,
                "max_tokens": 100,
                "stream": False,
            }
            future = endpoint.predict(
                request=launch.EndpointRequest(
                    args=request_body,
                    return_pickled=False,
                )
            )
            response = future.get()
            print(response)

            request_body = {
                "prompt": prompt,
                "temperature": 0.0,
                "max_tokens": 100,
                "stream": False,
                "decoding_regex_schema": regex,
            }
            future = endpoint.predict(
                request=launch.EndpointRequest(
                    args=request_body,
                    return_pickled=False,
                )
            )
            response = future.get()
            print(response)

    bundle_name = f"{BUNDLE_TAG}-{args.image_tag}"
    bundle_config = get_bundle_config(bundle_name, args.image_tag)

    if args.update_bundle:
        print(f"Deploying to launch env {LAUNCH_ENV}")
        endpoint = launch_client.get_model_endpoint(ENDPOINT_NAME).model_endpoint
        launch_client.create_model_bundle_from_runnable_image_v2(**bundle_config)
        launch_client.edit_model_endpoint(model_endpoint=endpoint, model_bundle=bundle_name)

        print(f"Model endpoint updated: {ENDPOINT_NAME}")

    elif args.redeploy:
        print(f"Deploying to launch env {LAUNCH_ENV}")
        # Delete current service
        try:
            launch_client.delete_model_endpoint(ENDPOINT_NAME)
        except AttributeError:
            pass
        # Wait for delete to happen successfully.
        sleep(10)
        launch_client.create_model_bundle_from_runnable_image_v2(**bundle_config)
        endpoint = launch_client.create_model_endpoint(
            model_bundle=bundle_name,
            **get_pytorch_service_params(ENDPOINT_NAME, ENDPOINT_TYPE),
        )
        print(f"VLLM endpoint launched: {ENDPOINT_NAME}")
