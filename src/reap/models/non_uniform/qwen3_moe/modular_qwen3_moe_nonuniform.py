# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen3 model."""

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.integrations import use_experts_implementation
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging
from transformers.utils.generic import OutputRecorder
from transformers.models.llama.modeling_llama import (
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaRMSNorm,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralForCausalLM,
    MixtralModel,
    MixtralPreTrainedModel,
    load_balancing_loss_func,
)
from transformers.models.qwen2_moe.modeling_qwen2_moe import (
    Qwen2MoeDecoderLayer,
    Qwen2MoeMLP,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from .configuration_qwen3_moe_nonuniform import Qwen3MoeConfig, NonUniformQwen3MoeConfig


logger = logging.get_logger(__name__)


class Qwen3MoeAttention(Qwen3Attention):  # This is the main diff with qwen2Moe!
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.layer_type
        self.sliding_window = getattr(config, "sliding_window", None)


class Qwen3MoeMLP(Qwen2MoeMLP):
    pass


@use_experts_implementation
class Qwen3MoeExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config, num_experts: Optional[int] = None):
        super().__init__()
        self.num_experts = config.num_experts if num_experts is None else int(num_experts)
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class Qwen3MoeTopKRouter(nn.Module):
    def __init__(self, config, num_experts: Optional[int] = None):
        super().__init__()
        self.num_experts = config.num_experts if num_experts is None else int(num_experts)
        self.top_k = min(int(config.num_experts_per_tok), int(self.num_experts))
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, num_experts: Optional[int] = None):
        super().__init__()
        self.num_experts = config.num_experts if num_experts is None else int(num_experts)
        self.experts = Qwen3MoeExperts(config, num_experts=self.num_experts)
        self.gate = Qwen3MoeTopKRouter(config, num_experts=self.num_experts)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class Qwen3MoeRMSNorm(LlamaRMSNorm):
    pass


class Qwen3MoeDecoderLayer(Qwen2MoeDecoderLayer):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        num_experts = config.get_num_experts(layer_idx) if hasattr(config, "get_num_experts") else config.num_experts
        if (layer_idx not in config.mlp_only_layers) and (
            num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(config, num_experts=num_experts)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)


class Qwen3MoePreTrainedModel(MixtralPreTrainedModel):
    _can_record_outputs = {
        "router_logits": OutputRecorder(Qwen3MoeTopKRouter, layer_name="mlp.gate", index=0),
        "hidden_states": Qwen3MoeDecoderLayer,
        "attentions": Qwen3MoeAttention,
    }


class Qwen3MoeModel(MixtralModel):
    pass


class Qwen3MoeForCausalLM(MixtralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3MoeModel(config)
        self.num_experts = config.num_experts

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3MoeForCausalLM

        >>> model = Qwen3MoeForCausalLM.from_pretrained("Qwen/Qwen3-MoE-15B-A2B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-MoE-15B-A2B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


class Qwen3MoeForSequenceClassification(LlamaForSequenceClassification):
    pass


class Qwen3MoeForTokenClassification(LlamaForTokenClassification):
    pass


class Qwen3MoeForQuestionAnswering(LlamaForQuestionAnswering):
    pass


class NonUniformQwen3MoePreTrainedModel(Qwen3MoePreTrainedModel):
    config_class = NonUniformQwen3MoeConfig


class NonUniformQwen3MoeModel(Qwen3MoeModel):
    config_class = NonUniformQwen3MoeConfig


class NonUniformQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    config_class = NonUniformQwen3MoeConfig


__all__ = [
    "Qwen3MoeForCausalLM",
    "Qwen3MoeForQuestionAnswering",
    "Qwen3MoeModel",
    "Qwen3MoePreTrainedModel",  # noqa: F822
    "Qwen3MoeForSequenceClassification",
    "Qwen3MoeForTokenClassification",
    "NonUniformQwen3MoeForCausalLM",
    "NonUniformQwen3MoeModel",
    "NonUniformQwen3MoePreTrainedModel",
]
