# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Modified HF GPT2 w/flash attention"""

import math
import os
from typing import Optional, Tuple, Union

import torch
from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from torch import nn
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2MLP, CausalLMOutputWithCrossAttentions, GPT2Attention, GPT2Block,
    GPT2LMHeadModel, GPT2Model, GPT2PreTrainedModel)


class GPT2FlashAttention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config=config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        if self.reorder_and_upcast_attn:
            raise ValueError('GPT2FlashAttention does not support reorder_and_upcast_attn')

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # rearrange to flash attention form
        key = rearrange(key, 'b h s d -> b s h d')
        value = rearrange(value, 'b h s d -> b s h d')
        query = rearrange(query, 'b h s d -> b s h d')

        #assert query.dtype in [torch.float16, torch.bfloat16], f"{query.dtype}"

        # stack
        qkv = torch.stack([query,key,value], dim=2)
        #qkv = torch.tensor(qkv,dtype=torch.bfloat16)
        assert qkv.dtype in [torch.float16, torch.bfloat16]

        # flash attention logic
        batch_size = qkv.shape[0]
        seqlen = qkv.shape[1]
        num_heads = qkv.shape[3]
        dk = qkv.shape[4]
        dk_per_head = int(dk)/int(num_heads)
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        max_s = seqlen
        cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=qkv.device)
        if self.training:
            attn_pdrop = 0.1
        else:
            attn_pdrop = 0.0
        softmax_scale = 1/float(math.sqrt(dk))
        output = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_seqlens, max_s, attn_pdrop,
            softmax_scale=softmax_scale, causal=True
        )
        output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        output = rearrange(output, 'b s h d -> b h s d')
        #output = torch.tensor(output, dtype=torch.float32)
        return output, None


class GPT2FlashBlock(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super(GPT2Block, self).__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2FlashAttention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2FlashAttention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)


class GPT2FlashModel(GPT2Model):
    def __init__(self, config):
        super(GPT2Model, self).__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2FlashBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class GPT2FlashLMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__(config)

        self.transformer = GPT2FlashModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

        # Special Case! When using the LMHeadModel, the weights of the self.lm_head and self.transformer.wte are tied.
        # This tying occurs inside the `self.post_init()` function call above.
        # This is a hurdle for FSDP because they need to be in the same FSDP block
        # These lines ensures that both modules stay together in the top-most block
        self.transformer._fsdp_wrap = False
        self.transformer.wte._fsdp_wrap = False
        self.lm_head._fsdp_wrap = False

    # Meta tensor param init fn
    def param_init_fn(self, module):
        if isinstance(module, GPT2LMHeadModel):
            module.post_init()

    # FSDP Wrap function
    def fsdp_wrap_fn(self, module):
        return isinstance(module, GPT2Block)

    # Activation Checkpointing
    def activation_checkpointing_fn(self, module):
        return isinstance(module, GPT2Block)
