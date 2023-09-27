from typing import Callable, List, Optional, Tuple, Union

import torch
from outlines.models.transformers import Tokenizer, Transformers
from outlines.text.generate import regex
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.sampling_params import SamplingParams


# A few utilities to mock outlines objects allowing us to leverage their create_proposal function
# directly to create logit masks for regex constrained sampling.
class TransformersTokenizerWrapper(Tokenizer):
    """Represents a tokenizer for models in the `transformers` library."""

    def __init__(self, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]):
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.eos_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id
            self.pad_token = self.tokenizer.pad_token

        self.vocabulary = self.tokenizer.get_vocab()

    def encode(
        self, prompt: Union[str, List[str]], **kwargs
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        kwargs["padding"] = True
        kwargs["return_tensors"] = "pt"
        output = self.tokenizer(prompt, **kwargs)
        return output["input_ids"], output["attention_mask"]

    def decode(self, token_ids: torch.LongTensor) -> List[str]:
        text = self.tokenizer.batch_decode(token_ids)
        return text

    def convert_token_to_string(self, token: str) -> str:
        string = self.tokenizer.convert_tokens_to_string([token])
        return string


class MockTransformersModel:
    def to(self, device: str):
        pass


def get_outlines_model(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> Transformers:
    outlines_tokenizer = TransformersTokenizerWrapper(tokenizer)
    outlines_model = Transformers(MockTransformersModel(), outlines_tokenizer)

    return outlines_model


# Creates a decoding function for a given llm and set of sampling parameters.
# If sampling parameters.decoding_regex_schema isn't set returns None.
def get_outlines_decoding_function(
    outlines_model: Transformers, sampling_params: SamplingParams, vocab_size
) -> Optional[Callable]:
    if sampling_params.decoding_regex_schema is None:
        return None

    outlines_program = regex(outlines_model, sampling_params.decoding_regex_schema)
    # It's possible for len(outlines_model.tokenizer.vocabulary) != vocab_size
    # when extra tokens have been added to the tokenizer but not tracked in the model.
    mock_logits = torch.zeros((1, len(outlines_model.tokenizer.vocabulary)), dtype=torch.double)

    # This decoding function takes the list of generated tokens and returns a logit bias mask
    # that forces only tokens that will match the regex to be generated. The mask is added to the
    # logits before the softmax is run so mask values for allowed tokens are 0 and mask values for
    # disallowed tokens are -inf. The biases are stored on CPU for now and will be moved to the
    # proper device during sampling.
    def decoding_function(generated_token_ids: List[int]) -> torch.Tensor:
        input_token_ids = torch.LongTensor([generated_token_ids])
        raw_biases = outlines_program.create_proposal(input_token_ids, mock_logits)[0].type(
            torch.HalfTensor
        )
        if vocab_size < len(raw_biases):
            # TODO: should we raise an error here?
            biases = raw_biases[:vocab_size]
        elif vocab_size > len(raw_biases):
            # We prevent generation for any tokens past the tokenizer's vocab length.
            biases = torch.zeros(vocab_size, dtype=torch.half) + float("-Inf")
            biases[: len(raw_biases)] = raw_biases
        else:
            biases = raw_biases
        return biases

    return decoding_function
