"""
mistral_gpt2.py

Custom Implementation of the GPT-2 LM-Head Model (and auxiliary classes) with support for adaptive/custom number of
gradient checkpoints (for fine-grained tweaking of memory footprint vs. speed).

Reference: https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py
"""
import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import Attention, Block


# Nest Overwatch under root `mistral` logger, inheriting formatting!
overwatch = logging.getLogger("mistral.models.gpt2_gc")


class MistralGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config: GPT2Config, reorder_attn: bool = True, upcast_attn: bool = True):
        super().__init__(config)
        self.reorder_attn, self.upcast_attn = reorder_attn, upcast_attn

    # @MERCURY =>> Reconfigure GPT2LMHead to take custom, partial checkpoint model instance!
    def create_checkpointed_model(self, gc_checkpoint_every: int):
        # Reinitalize GPT-2 Model w/ Custom GC Wrapper
        self.transformer = MistralGPT2Model(self.config, gc_checkpoint_every, self.reorder_attn, self.upcast_attn)

    # @MERCURY =>> Reconfigure GPT2LMHead to Initialize Standard (non-checkpointed) model instance!
    def create_model(self):
        # Reinitialize Custom GPT-2 Model
        self.transformer = MistralGPT2Model(
            self.config, gc_checkpoint_every=-1, reorder_attn=self.reorder_attn, upcast_attn=self.upcast_attn
        )


class MistralGPT2Model(GPT2Model):
    # @MERCURY =>> GPT-2 Model Instance now takes `gc_checkpoint_every` parameter.
    def __init__(self, config: GPT2Config, gc_checkpoint_every: int, reorder_attn: bool, upcast_attn: bool):
        super().__init__(config)
        self.h = nn.ModuleList(
            [
                MistralGPT2Block(
                    config.n_ctx, config, i + 1, scale=True, reorder_attn=reorder_attn, upcast_attn=upcast_attn
                )
                for i in range(config.n_layer)
            ]
        )
        self.init_weights()

        if getattr(self.config, "gradient_checkpointing", False):
            assert 1 <= gc_checkpoint_every <= len(self.h)
            self.gc_checkpoint_every = gc_checkpoint_every

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # @MERCURY =>> Single line change, `and (i % self.gc_checkpoint_every) == 0` --> partial-checkpointing!
            if (
                getattr(self.config, "gradient_checkpointing", False)
                and self.training
                and (i % self.gc_checkpoint_every) == 0
            ):
                if use_cache:
                    overwatch.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )

            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class MistralGPT2Attention(Attention):
    def __init__(
        self, nx, n_ctx, config, layer_num, scale=False, is_cross_attention=False, reorder_attn=True, upcast_attn=True
    ):
        super().__init__(nx, n_ctx, config, scale, is_cross_attention)

        self.activation_stats = {
            "attention_weight_max": None,
            "attention_weight_min": None,
        }
        assert layer_num > 0
        self.layer_num = layer_num

        # Numerical Stability
        self.reorder_attn, self.upcast_attn = reorder_attn, upcast_attn

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        """
        Taken from:
            https://github.com/huggingface/transformers/blob/v4.5.0/src/transformers/models/gpt2/modeling_gpt2.py#L167

        We log extra statistics about the attention weights!
        """
        # @MERCURY =>> Reorder Scaled Dot-Product Attention Computation, Upcast to FP32
        # Q :: [bsz, num_heads, seq_len, dk], K :: [bsz, num_heads, dk, seq_len]
        if self.scale:
            # Get QKV Dimensions
            bsz, num_heads, seq_len, dk = q.size()

            # @MERCURY =>> Scale by SQRT(head_dim)*layer_number. Taken from MegatronLM.
            # Compute Scale Factor
            scale_factor = 1 / ((float(v.size(-1)) ** 0.5) * self.layer_num)

            if self.reorder_attn:
                # Preallocate Scaled Dot-Product Tensor
                w = torch.empty(
                    bsz * num_heads,
                    seq_len,
                    seq_len,
                    dtype=q.dtype,
                    device=torch.cuda.current_device(),
                )

                # Upcasting --> Disable autocast AND manually call .float()
                if self.upcast_attn:
                    # Reorder via `baddbmm` Time (Scale K by 1 / root(dk) first!)
                    with autocast(enabled=False):
                        q, k = q.reshape(-1, seq_len, dk), k.reshape(-1, dk, seq_len)
                        w = torch.baddbmm(
                            w.float(),
                            q.float(),
                            k.float(),
                            beta=0.0,
                            alpha=scale_factor,
                        )
                        w = w.reshape(bsz, num_heads, seq_len, seq_len)

                # No Upcasting
                else:
                    q, k = q.reshape(-1, seq_len, dk), k.reshape(-1, dk, seq_len)
                    w = torch.baddbmm(w, q, k, beta=0.0, alpha=scale_factor)
                    w = w.reshape(bsz, num_heads, seq_len, seq_len)

            else:
                # Upcasting --> Disable autocast AND manually call .float()
                if self.upcast_attn:
                    with autocast(enabled=False):
                        w = torch.matmul(q.float(), k.float())
                        w *= scale_factor

                # No Upcasting
                else:
                    w = torch.matmul(q, k)
                    w *= scale_factor

        else:
            w = torch.matmul(q, k)

        # Add extra logging of the attention weight
        with torch.no_grad():
            self.activation_stats["attention_weight_max"] = w.max().item()
            self.activation_stats["attention_weight_min"] = w.min().item()

        nd, ns = w.size(-2), w.size(-1)
        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            mask = self.bias[:, :, ns - nd : ns, :ns]
            w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)

        # @MERCURY =>> Downcast (if necessary) back to V dtype (fp16 if mixed-precision)!
        # Note: This is a No-Op if Upcasting is disabled...
        w = w.type(v.dtype)

        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = (torch.matmul(w, v),)
        if output_attentions:
            outputs += (w,)
        return outputs


class MistralGPT2Block(Block):
    def __init__(self, n_ctx, config, layer_num, scale=False, reorder_attn=True, upcast_attn=True):
        super().__init__(n_ctx, config, scale)
        hidden_size = config.n_embd
        self.attn = MistralGPT2Attention(
            hidden_size, n_ctx, config, layer_num, scale=scale, reorder_attn=reorder_attn, upcast_attn=upcast_attn
        )
