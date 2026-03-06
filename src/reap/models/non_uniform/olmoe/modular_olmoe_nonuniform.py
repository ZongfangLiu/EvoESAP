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
"""PyTorch OLMoE model."""

from collections.abc import Callable
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, is_grouped_mm_available, logging
from transformers.utils.generic import OutputRecorder
from transformers.models.gemma.modeling_gemma import GemmaMLP
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MixtralModel

from .configuration_olmoe_nonuniform import OlmoeConfig, NonUniformOlmoeConfig


logger = logging.get_logger(__name__)


class OlmoeRMSNorm(LlamaRMSNorm):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__(hidden_size, eps)


class OlmoeRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class OlmoeMLP(GemmaMLP):
    pass


class OlmoeAttention(LlamaAttention):
    def __init__(self, config: OlmoeConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.q_norm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.k_norm = OlmoeRMSNorm(
            (config.hidden_size // config.num_attention_heads) * config.num_key_value_heads, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        if self.config.clip_qkv is not None:  # Diff with llama
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        query_states = query_states.view(*hidden_shape).transpose(1, 2)
        key_states = key_states.view(*hidden_shape).transpose(1, 2)
        value_states = value_states.view(*hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self.config, "sliding_window", None),  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class OlmoeExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config: OlmoeConfig, num_experts: Optional[int] = None):
        super().__init__()
        self.num_experts = config.num_experts if num_experts is None else int(num_experts)
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
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
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class OlmoeTopKRouter(nn.Module):
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


class OlmoeSparseMoeBlock(nn.Module):
    def __init__(self, config, num_experts: Optional[int] = None):
        super().__init__()
        self.num_experts = config.num_experts if num_experts is None else int(num_experts)
        self.gate = OlmoeTopKRouter(config, num_experts=self.num_experts)
        self.experts = OlmoeExperts(config, num_experts=self.num_experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        _, top_k_weights, top_k_index = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states, top_k_index, top_k_weights).reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states


class OlmoeDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: OlmoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.self_attn = OlmoeAttention(config=config, layer_idx=layer_idx)
        num_experts = config.get_num_experts(layer_idx) if hasattr(config, "get_num_experts") else config.num_experts
        self.mlp = OlmoeSparseMoeBlock(config, num_experts=num_experts)
        self.input_layernorm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


@auto_docstring
class OlmoePreTrainedModel(PreTrainedModel):
    config: OlmoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OlmoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(OlmoeTopKRouter, index=0),
        "hidden_states": OlmoeDecoderLayer,
        "attentions": OlmoeAttention,
    }
    _can_compile_fullgraph = (
        is_grouped_mm_available()
    )  # https://huggingface.co/docs/transformers/experts_interface#torchcompile
    _supports_attention_backend = True

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, OlmoeExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, OlmoeTopKRouter):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


@auto_docstring
class OlmoeModel(MixtralModel):
    def __init__(self, config: OlmoeConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [OlmoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = OlmoeRotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(  # diff with mixtral: no sliding
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class OlmoeForCausalLM(MixtralForCausalLM, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = OlmoeModel(config)
        self.num_experts = config.num_experts

    def forward(self, **super_kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OlmoeForCausalLM

        >>> model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924")
        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        'Hey, are you conscious? Can you talk to me?\nI’m not sure if you’re conscious of this, but I’m'
        ```
        """
        return super().forward(**super_kwargs)


class NonUniformOlmoePreTrainedModel(OlmoePreTrainedModel):
    config_class = NonUniformOlmoeConfig


class NonUniformOlmoeModel(OlmoeModel):
    config_class = NonUniformOlmoeConfig


class NonUniformOlmoeForCausalLM(OlmoeForCausalLM):
    config_class = NonUniformOlmoeConfig


__all__ = [
    "OlmoeForCausalLM",
    "OlmoeModel",
    "OlmoePreTrainedModel",
    "NonUniformOlmoeForCausalLM",
    "NonUniformOlmoeModel",
    "NonUniformOlmoePreTrainedModel",
]
