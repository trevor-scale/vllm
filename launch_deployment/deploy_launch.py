"""Contains code for working with launch endpoint (Checking endpoint, testing, deploying, etc)"""
# python launch_deployment/deploy_launch.py --image_tag vllm-63aeeb78b835232d7f0c7bcbfe1f7d364b83bb06 --redeploy
import argparse
import json
import os
from time import sleep
from typing import Dict, List, Optional

import launch
from launch_internal import get_launch_client
from pydantic import BaseModel

LAUNCH_ENV = os.getenv("LAUNCH_ENV", "prod")  # or "training"

BUNDLE_TAG = "vllm-guided-decoding"
ENDPOINT_TYPE = "async"
FEDERAL_ID = "63bf16fbc65a8abb84ad8619"

launch_client = get_launch_client(api_key=FEDERAL_ID, env=LAUNCH_ENV, v1=True)

TEST_REGEXES = [
    r"\s*([Yy]es|[Nn]o|[Nn]ever|[Aa]lways)",
    '{\n  "gender": ("m"|"f"),\n  "height_inches": (0|[1-9][0-9]*),\n  "weight_lbs": (0|[1-9][0-9]*)\n}',
]
TEST_PROMPTS = [
    "Is 1+1=2? ",
    "Write a json blob describing a tennis player including the fields gender (m/f), height (inches) and weight (lbs): ",
]


class RequestModel(BaseModel):
    prompt: str
    temperature: float
    max_tokens: int
    stream: bool
    decoding_regex_schema: Optional[str]
    token_healing: bool


class ResponseModel(BaseModel):
    text: str
    count_prompt_tokens: int
    count_output_tokens: int
    log_probs: Dict
    tokens: List[str]


def load_config(model_name: str):
    with open(os.path.join(os.path.dirname(__file__), "model_configs.json"), "r") as f:
        config = json.load(f)
    return config[model_name]


def test_endpoint(endpoint: launch.AsyncEndpoint):
    for prompt, regex in zip(TEST_PROMPTS, TEST_REGEXES):
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
        print("WITHOUT REGEX DECODING:")
        if response.status == "SUCCESS":
            print(json.loads(response.result)["text"])
        else:
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
        print("WITH REGEX DECODING:")
        if response.status == "SUCCESS":
            print(json.loads(response.result)["text"])
        else:
            print(response)


def get_bundle_config(
    bundle_name: str,
    checkpoint_path: str,
    image_tag: str,
    num_shards: int,
    max_num_batched_tokens: int,
) -> Dict:
    base_path = checkpoint_path.split("/")[-1]
    subcommands = [
        f"./s5cmd cp {checkpoint_path} .",
        f"mkdir -p model_files",
        f"tar --no-same-owner -xf {base_path} -C model_files",
        f'python -m vllm_server_launch --model model_files --tensor-parallel-size {num_shards} --port 5005 --max-num-batched-tokens {max_num_batched_tokens} --host "::"',
    ]

    return {
        "model_bundle_name": bundle_name,
        "request_schema": RequestModel,
        "response_schema": ResponseModel,
        "repository": "fractal-chat-service",
        "tag": image_tag,
        "command": [
            "/bin/bash",
            "-c",
            ";".join(subcommands),
        ],
        "readiness_initial_delay_seconds": 20,
        "env": {},
    }


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("model_name")
    argparser.add_argument("--endpoint_test", action="store_true")
    argparser.add_argument("--redeploy", action="store_true")
    argparser.add_argument("--update_bundle", action="store_true")
    argparser.add_argument("--image_tag", default="")
    argparser.add_argument("--check_endpoint", action="store_true")

    args = argparser.parse_args()
    if args.check_endpoint:
        for endpoint in launch_client.list_model_endpoints():
            print(endpoint.model_endpoint)
    model_name = args.model_name
    config = load_config(model_name)
    bundle_name = f"{BUNDLE_TAG}-{model_name}"
    endpoint_name = f"{bundle_name}-{ENDPOINT_TYPE}-service"

    if args.endpoint_test:
        endpoint = launch_client.get_model_endpoint(endpoint_name)
        test_endpoint(endpoint)

    bundle_config = get_bundle_config(
        bundle_name,
        config["checkpoint_path"],
        args.image_tag,
        config["num_shards"],
        config["max_num_batched_tokens"],
    )

    if args.update_bundle:
        print(f"Deploying to launch env {LAUNCH_ENV}")
        endpoint = launch_client.get_model_endpoint(endpoint_name).model_endpoint
        launch_client.create_model_bundle_from_runnable_image_v2(**bundle_config)
        launch_client.edit_model_endpoint(model_endpoint=endpoint, model_bundle=bundle_name)

        print(f"Model endpoint updated: {endpoint_name}")

    elif args.redeploy:
        print(f"Deploying to launch env {LAUNCH_ENV}")
        # Delete current service
        try:
            launch_client.delete_model_endpoint(endpoint_name)
        except AttributeError:
            pass
        # Wait for delete to happen successfully.
        sleep(10)
        service_params = config["service_params"]
        service_params["endpoint_type"] = ENDPOINT_TYPE
        service_params["endpoint_name"] = endpoint_name
        launch_client.create_model_bundle_from_runnable_image_v2(**bundle_config)
        endpoint = launch_client.create_model_endpoint(
            model_bundle=bundle_name,
            **service_params,
        )
        print(f"VLLM endpoint launched: {endpoint_name}")
