"""Convert datasets to transformers BatchEncoded or vLLM TokensPrompt formats.

We follow the OpenAI spec for conversational datasets.

ie..,
messages = [
    {"role": "system", "content": "You are AGI"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "What is my purpose?"},
]

Includes the ability to select from specific categories within the dataset and convert
the dataset into either a language modelling dataset with attention applied to every
token or a prompt-completion dataset for training on completions only with SFTTrainer.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import random
import logging
from functools import partial


import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, BatchEncoding
from vllm import TokensPrompt


logger = logging.getLogger(__name__)


# --- Base Dataset Processors --------------------------------------------------


class BaseDatasetProcessor(ABC):
    category_field: str = "category"
    text_field: str = "text"
    completion_field: str = "completion"
    prompt_field: str = "prompt"
    messages_field: str = "messages"
    all_categories_label: str = "all"

    def __init__(
        self,
        dataset: Dataset | DatasetDict,
        tokenizer: AutoTokenizer,
        pack_samples: bool = True,
        max_input_len: int | None = None,
        split: str | None = None,
        split_by_category: bool = True,
        return_vllm_tokens_prompt: bool = False,
        truncate: bool = False,
        select_only_categories: list[str] | str | None = None,
    ):
        """Defines base functionality for all Dataset Processors.

        Args:
            dataset (Dataset | DatasetDict): _description_
            tokenizer (AutoTokenizer): _description_
            split (str | None, optional): _description_. Defaults to None.
            split_by_category (bool, optional): _description_. Defaults to True.
            return_vllm_tokens_prompt (bool, optional): If True, will return
                TokensPrompt objects instead of BatchEncoding. Defaults to False
            truncate (bool, optional): If True, will truncate the samples from
                the dataset to the max_input_len instead of skipping them.

        """
        if isinstance(dataset, DatasetDict):
            if split is None:
                split = list(dataset.keys())[0]
                logging.warning(
                    f"Using split '{split}' as default for dataset. Available "
                    f"splits: {list(dataset.keys())}",
                )
            dataset = dataset[split]
        if max_input_len is None:
            max_input_len = tokenizer.model_max_length
            logger.warning(
                f"max_input_len is set to {max_input_len} as per tokenizer's "
                f"model_max_length. This will be used for truncation.",
            )
        self.pack_samples = pack_samples
        self.max_input_len = max_input_len
        self.dataset = dataset
        self._mapped_dataset = None
        self.tokenizer = tokenizer
        self.split_by_category = split_by_category
        self.return_vllm_tokens_prompt = return_vllm_tokens_prompt
        self.truncate = truncate
        self.categories = self.get_categories()
        if isinstance(select_only_categories, str):
            select_only_categories = [select_only_categories]
        self.select_only_categories = select_only_categories
        if self.select_only_categories:
            logger.warning(
                "select_only_categories is not None but split_by_category "
                "was False. Setting split_by_category to True and processing "
                f"categories: {self.select_only_categories}"
            )
            self.split_by_category = True
            if self.category_field not in self.dataset.column_names:
                raise RuntimeError(
                    f"Category field '{self.category_field}' not found in dataset. "
                    "Please ensure the dataset has a category field.",
                )
            for category in self.select_only_categories:
                if category not in self.categories:
                    raise RuntimeError(
                        f"Category '{category}' not found in dataset. "
                        "Please ensure the dataset has the specified categories.",
                    )

    @staticmethod
    @abstractmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        """Map a row of the dataset to the desired output format.

        EG., map "prompts" and "completions" to "messages" for chat datasets.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses.",
        )

    @abstractmethod
    def _encode_sample(self, sample: str) -> torch.Tensor:
        """Encode a str sample from the desired category of the dataset into
        tokens.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses.",
        )

    def get_processed_dataset(
        self,
        samples_per_category: int,
        return_loss_mask: bool = False,
    ) -> dict[str, list[TokensPrompt]] | dict[str, list[BatchEncoding]] | dict[str, list[dict]]:
        """Get requests for each category in the dataset."""
        if self._mapped_dataset is None:
            self._mapped_dataset = self.dataset.map(self._map_fn)
        if self.split_by_category:
            categories = (
                self.categories
                if self.select_only_categories is None
                else self.select_only_categories
            )
            return {
                c: self._process_samples_for_category(
                    c, samples_per_category, return_loss_mask=return_loss_mask
                )
                for c in categories
            }
        else:
            return {
                self.all_categories_label: self._process_samples_for_category(
                    self.all_categories_label,
                    samples_per_category,
                    return_loss_mask=return_loss_mask,
                ),
            }

    def get_categories(self) -> list[str]:
        """Get the unique categories in the dataset."""
        if self.category_field is None:
            logger.warning(
                "No category field specified for dataset, returning 'all' category."
            )
            return ["all"]
        return self.dataset.unique(self.category_field)

    def _process_samples_for_category(
        self,
        category: str,
        samples_per_category: int,
        return_loss_mask: bool = False,
    ) -> list[TokensPrompt] | list[BatchEncoding]:
        if category != self.all_categories_label:
            category_dataset = self._mapped_dataset.filter(
                lambda sample: sample[self.category_field] == category,
            )
        else:
            category_dataset = self._mapped_dataset
            category = self.all_categories_label

        if self.pack_samples:
            return self._process_samples_for_category_packed(
                category, samples_per_category, category_dataset, return_loss_mask
            )
        else:
            return self._process_samples_for_category_unpacked(
                category, samples_per_category, category_dataset, return_loss_mask
            )

    def _process_samples_for_category_unpacked(
        self,
        category: str,
        samples_per_category: int,
        category_dataset: Dataset,
        return_loss_mask: bool = False,
    ) -> list[TokensPrompt] | list[BatchEncoding]:
        processed_samples = []
        sampled = []  # sample without replacement
        while len(processed_samples) < samples_per_category:
            if len(sampled) >= len(category_dataset):
                logger.warning(
                    f"Not enough samples in category '{category}' to reach "
                    f"{samples_per_category} samples. Only {len(sampled)} "
                    "samples were processed.",
                )
                break
            sample_idx = random.randint(0, len(category_dataset) - 1)
            if sample_idx in sampled:
                continue
            sampled.append(sample_idx)
            sample = category_dataset[sample_idx]
            labels_mask = None
            if return_loss_mask and hasattr(self, "_encode_sample_with_mask"):
                encoded_sample, labels_mask = self._encode_sample_with_mask(sample)
            else:
                encoded_sample = self._encode_sample(sample)
                if return_loss_mask:
                    labels_mask = torch.ones_like(encoded_sample)
            if encoded_sample.shape[-1] > self.max_input_len:
                if self.truncate:
                    encoded_sample = encoded_sample[:, : self.max_input_len]
                    if labels_mask is not None:
                        labels_mask = labels_mask[:, : self.max_input_len]
                else:
                    continue

            if self.return_vllm_tokens_prompt:
                encoded_sample = TokensPrompt(
                    prompt_token_ids=encoded_sample[0, :-1].tolist()
                )
            elif return_loss_mask:
                encoded_sample = {
                    "input_ids": encoded_sample,
                    "attention_mask": torch.ones_like(encoded_sample),
                    "labels_mask": labels_mask if labels_mask is not None else torch.ones_like(encoded_sample),
                }
            processed_samples.append(encoded_sample)
        return processed_samples

    def _process_samples_for_category_packed(
        self,
        category: str,
        samples_per_category: int,
        category_dataset: Dataset,
        return_loss_mask: bool = False,
    ) -> list[TokensPrompt] | list[BatchEncoding]:
        processed_samples = []
        sampled = []
        while len(processed_samples) < samples_per_category:
            if len(sampled) >= len(category_dataset):
                logger.warning(
                    f"Not enough samples in category '{category}' to reach "
                    f"{samples_per_category} samples. Only {len(sampled)} "
                    "samples were processed.",
                )
                break
            seq = torch.zeros((1, self.max_input_len), dtype=torch.long)
            labels_seq = torch.zeros_like(seq) if return_loss_mask else None
            seq_idx = 0
            while seq_idx < self.max_input_len:
                if len(sampled) >= len(category_dataset):
                    logger.warning(
                        f"Not enough samples to pack last sequence to max_input_len."
                    )
                    break
                sample_idx = random.randint(0, len(category_dataset) - 1)
                if sample_idx in sampled:
                    continue
                sampled.append(sample_idx)
                sample = category_dataset[sample_idx]
                labels_mask = None
                if return_loss_mask and hasattr(self, "_encode_sample_with_mask"):
                    encoded_sample, labels_mask = self._encode_sample_with_mask(sample)
                else:
                    encoded_sample = self._encode_sample(sample)  # shape (batch, seq)
                    if return_loss_mask:
                        labels_mask = torch.ones_like(encoded_sample)
                end_seq = seq_idx + encoded_sample.shape[-1]
                if end_seq > self.max_input_len:
                    encoded_sample = encoded_sample[:, : (self.max_input_len - seq_idx)]
                    if labels_mask is not None:
                        labels_mask = labels_mask[:, : (self.max_input_len - seq_idx)]
                    end_seq = self.max_input_len
                seq[:, seq_idx:end_seq] = encoded_sample
                if labels_mask is not None and labels_seq is not None:
                    labels_seq[:, seq_idx:end_seq] = labels_mask
                seq_idx = end_seq + 1
            if self.return_vllm_tokens_prompt:
                encoded_sample = TokensPrompt(
                    prompt_token_ids=seq[0, :-1].tolist()  # -1 for vLLM.LLM.generate
                )
            else:
                if return_loss_mask:
                    encoded_sample = {
                        "input_ids": seq,
                        "attention_mask": torch.ones_like(seq),
                        "labels_mask": labels_seq if labels_seq is not None else torch.ones_like(seq),
                    }
                else:
                    encoded_sample = seq
            processed_samples.append(encoded_sample)
        return processed_samples


class ChatDatasetProcessor(BaseDatasetProcessor):
    def _encode_sample(self, sample: str) -> torch.Tensor:
        chat_sample = self.tokenizer.apply_chat_template(
            sample[self.messages_field],
            add_generation_prompt=False,
            tokenize=False,
        )
        return self.tokenizer(
            chat_sample,
            truncation=self.truncate,
            max_length=self.tokenizer.model_max_length if self.truncate else None,
            return_tensors="pt",
        )["input_ids"]

    def _encode_sample_with_mask(self, sample: dict[str, any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tokenized chat sample plus a mask over assistant tokens for loss."""
        messages = sample[self.messages_field]
        chat_sample = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )
        tokenized = self.tokenizer(
            chat_sample,
            truncation=self.truncate,
            max_length=self.tokenizer.model_max_length if self.truncate else None,
            return_tensors="pt",
        )["input_ids"]
        total_len = tokenized.shape[-1]
        loss_mask = torch.zeros_like(tokenized)

        prev_len = 0
        for idx, message in enumerate(messages):
            partial = self.tokenizer.apply_chat_template(
                messages[: idx + 1],
                add_generation_prompt=False,
                tokenize=False,
            )
            partial_ids = self.tokenizer(
                partial,
                truncation=self.truncate,
                max_length=self.tokenizer.model_max_length if self.truncate else None,
                return_tensors="pt",
            )["input_ids"]
            curr_len = partial_ids.shape[-1]
            start = min(prev_len, total_len)
            end = min(curr_len, total_len)
            if message.get("role") == "assistant":
                loss_mask[:, start:end] = 1
            prev_len = curr_len
        return tokenized, loss_mask

    def get_llmcompressor_dataset(self) -> Dataset:
        """Get the mapped dataset without tokenization applied."""

        def chat_template_fn(sample: dict[str, any]) -> dict[str, any]:
            """Apply chat template to the sample."""
            chat_sample = self.tokenizer.apply_chat_template(
                sample[self.messages_field],
                add_generation_prompt=False,
                tokenize=False,
            )
            return {"text": chat_sample}

        if self._mapped_dataset is None:
            self._mapped_dataset = self.dataset.map(self._map_fn)

        return self._mapped_dataset.map(chat_template_fn)


class LMDatasetProcessor(BaseDatasetProcessor):
    def _encode_sample(self, sample: str) -> torch.Tensor:
        return self.tokenizer(
            sample[self.text_field],
            truncation=self.truncate,
            max_length=self.tokenizer.model_max_length if self.truncate else None,
            return_tensors="pt",
        )["input_ids"]

    def get_llmcompressor_dataset(self) -> Dataset:
        """Get the mapped dataset without tokenization applied."""

        if self._mapped_dataset is None:
            self._mapped_dataset = self.dataset.map(self._map_fn)

        return self._mapped_dataset


### --- Concrete Implementations -----------------------------------------------


class CodeFeedbackChatDataset(ChatDatasetProcessor):
    category_field: str = "lang"
    messages_field: str = "text_fieldmessages"

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample


class TuluSFTMixtureChatDataset(ChatDatasetProcessor):
    category_field: str = "source"

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample

class PersonasMathChatDataset(ChatDatasetProcessor):
    """Dataset for Tulu-3 SFT Personas Math."""
    
    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample

class WildChatSFTMixtureChatDataset(ChatDatasetProcessor):
    category_field: str = "langauge"

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample


class MmluChatDataset(ChatDatasetProcessor):
    category_field: str = "subject"

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"{sample['question']} "
                        f"Choose from the following options: {sample['choices']}"
                    ),
                },
            ],
        }


class MagicoderEvolInstructChatDataset(ChatDatasetProcessor):
    """Dataset for Magicoder-Evol-Instruct-110K."""

    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return {
            "messages": [
                {"role": "user", "content": sample["instruction"]},
                {"role": "assistant", "content": sample["response"]},
            ],
        }


class C4LMDataset(LMDatasetProcessor):
    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample


class CodeAlpacaChatDataset(ChatDatasetProcessor):
    """Dataset for evol-codealpaca-v1."""

    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return {
            "messages": [
                {"role": "user", "content": sample["instruction"]},
                {"role": "assistant", "content": sample["output"]},
            ],
        }

class WritingPromptsChatDataset(ChatDatasetProcessor):
    """Dataset for WritingPrompts_curated."""
    
    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return {
            "messages": [
                {"role": "user", "content": f"Please write a creative story using the following writing prompt:\n\n {sample['prompt']}"},
                {"role": "assistant", "content": sample["body"]},
            ],
        }



DATASET_REGISTRY: dict[str, BaseDatasetProcessor] = {
    "m-a-p/CodeFeedback-Filtered-Instruction": CodeFeedbackChatDataset,
    "allenai/tulu-3-sft-mixture": TuluSFTMixtureChatDataset,
    "cais/mmlu": MmluChatDataset,
    "ise-uiuc/Magicoder-Evol-Instruct-110K": MagicoderEvolInstructChatDataset,
    "allenai/c4": C4LMDataset,
    "theblackcat102/evol-codealpaca-v1": CodeAlpacaChatDataset,
    "euclaise/WritingPrompts_curated": WritingPromptsChatDataset,
    "allenai/tulu-3-sft-personas-math": PersonasMathChatDataset,
}
