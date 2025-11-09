from typing import List, Optional, Tuple, Union
import torch

from transformers import Qwen3Model, Qwen3ForCausalLM, Qwen3PreTrainedModel, Qwen3Config
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RotaryEmbedding,
)
from torch import nn
from transformers.utils import logging
from .attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

from peft import PeftModel

logger = logging.get_logger(__name__)


class ModifiedQwen3Attention(Qwen3Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


# class ModifiedQwen3FlashAttention2(Qwen3FlashAttention2):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.is_causal = False


# class ModifiedQwen3SdpaAttention(Qwen3SdpaAttention):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.is_causal = False


# QWEN3_ATTENTION_CLASSES = {
#     "eager": ModifiedQwen3Attention,
#     "flash_attention_2": ModifiedQwen3FlashAttention2,
#     "sdpa": ModifiedQwen3SdpaAttention,
# }


class ModifiedQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = ModifiedQwen3Attention(
            config=config, layer_idx=layer_idx
        )

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class Qwen3BiModel(Qwen3Model):
    _no_split_modules = ["ModifiedQwen3DecoderLayer"]

    def __init__(self, config: Qwen3Config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                ModifiedQwen3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


class Qwen3BiForMNTP(Qwen3ForCausalLM):
    def __init__(self, config):
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = Qwen3BiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # getter for PEFT model
    def get_model_for_peft(self):
        return self.model

    # setter for PEFT model
    def set_model_for_peft(self, model: PeftModel):
        self.model = model

    # save the PEFT model
    def save_peft_model(self, path):
        self.model.save_pretrained(path)
