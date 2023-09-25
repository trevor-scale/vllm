_SUPPORTED_MODEL_NAMES = {
      
}
def create_vllm_bundle(
        model_name: str = "llama-2-7b-chat",
        framework_image_tag: str,
        endpoint_unique_name: str,
        num_shards: int,
    ):
        command = []

        max_num_batched_tokens = 2560  # vLLM's default
        if "llama-2" in model_name:
            max_num_batched_tokens = 4096  # Need to be bigger than model's context window

        subcommands = []
        final_weights_folder = _SUPPORTED_MODEL_NAMES[model_name]

        subcommands.append(
            f"python -m vllm_server --model {final_weights_folder} --tensor-parallel-size {num_shards} --port 5005 --max-num-batched-tokens {max_num_batched_tokens}"
        )

        command = [
            "/bin/bash",
            "-c",
            ";".join(subcommands),
        ]

        return (
            await self.create_model_bundle_use_case.execute(
                user,
                CreateModelBundleV2Request(
                    name=endpoint_unique_name,
                    schema_location="TBA",
                    flavor=StreamingEnhancedRunnableImageFlavor(
                        flavor=ModelBundleFlavorType.STREAMING_ENHANCED_RUNNABLE_IMAGE,
                        repository=hmi_config.vllm_repository,
                        tag=framework_image_tag,
                        command=command,
                        streaming_command=command,
                        protocol="http",
                        readiness_initial_delay_seconds=10,
                        healthcheck_route="/health",
                        predict_route="/predict",
                        streaming_predict_route="/stream",
                        env={},
                    ),
                    metadata={},
                ),
                do_auth_check=False,
                # Skip auth check because llm create endpoint is called as the user itself,
                # but the user isn't directly making the action. It should come from the fine tune
                # job.
            )
        ).model_bundle_id