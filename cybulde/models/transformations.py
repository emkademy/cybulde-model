import os

from abc import ABC, abstractmethod

from torch import Tensor
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase

from cybulde.utils.io_utils import is_dir, is_file, translate_gcs_dir_to_local


class Transformation(ABC):
    @abstractmethod
    def __call__(self, texts: list[str]) -> BatchEncoding:
        ...

    @abstractmethod
    def decode(self, input_ids: Tensor, skip_special_tokens: bool = False) -> list[str]:
        ...


class HuggingFaceTokenizationTransformation(Transformation):
    def __init__(self, pretrained_tokenizer_name_or_path: str, max_sequence_length: int) -> None:
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.tokenizer = self.get_tokenizer(pretrained_tokenizer_name_or_path)

    def __call__(self, texts: list[str]) -> BatchEncoding:
        output: BatchEncoding = self.tokenizer.batch_encode_plus(
            texts, truncation=True, padding=True, return_tensors="pt", max_length=self.max_sequence_length
        )
        return output

    def decode(self, input_ids: Tensor, skip_special_tokens: bool = False) -> list[str]:
        output: list[str] = self.tokenizer.batch_decode(input_ids, skip_special_tokens=skip_special_tokens)
        return output

    def get_tokenizer(self, pretrained_tokenizer_name_or_path: str) -> PreTrainedTokenizerBase:
        if is_dir(pretrained_tokenizer_name_or_path):
            tokenizer_dir = translate_gcs_dir_to_local(pretrained_tokenizer_name_or_path)
        elif is_file(pretrained_tokenizer_name_or_path):
            pretrained_tokenizer_name_or_path = translate_gcs_dir_to_local(pretrained_tokenizer_name_or_path)
            tokenizer_dir = os.path.dirname(pretrained_tokenizer_name_or_path)
        else:
            tokenizer_dir = pretrained_tokenizer_name_or_path

        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(tokenizer_dir)
        return tokenizer
