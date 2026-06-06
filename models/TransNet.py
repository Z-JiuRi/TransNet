import copy
import math
import warnings
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_
from torch.nn.init import xavier_uniform_
from torch.nn.parameter import Parameter

__all__ = ["transnet"]

Tensor = torch.Tensor


def original_scaled_dot_product_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    _, _, embed_dim = q.shape
    q = q / math.sqrt(embed_dim)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = F.softmax(attn, dim=-1)
    if dropout_p:
        attn = F.dropout(attn, p=dropout_p)
    out = torch.bmm(attn, v)

    return out, attn


def original_in_projection_packed(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
):
    embed_dim = q.size(-1)
    if k is v:
        if q is k:
            return F.linear(q, w, b).chunk(3, dim=-1)

        w_q, w_kv = w.split([embed_dim, embed_dim * 2])
        if b is None:
            b_q = b_kv = None
        else:
            b_q, b_kv = b.split([embed_dim, embed_dim * 2])
        return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)

    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def original_multi_head_attention_forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Optional[Tensor],
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    tgt_len, batch_size, embed_dim = query.shape
    src_len, _, _ = key.shape
    head_dim = embed_dim // num_heads
    q, k, v = original_in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, (
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
            )

        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (batch_size * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                )
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    q = q.contiguous().view(tgt_len, batch_size * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, batch_size * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, batch_size * num_heads, head_dim).transpose(0, 1)
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (batch_size, src_len), (
            f"expecting key_padding_mask shape of {(batch_size, src_len)}, but got {key_padding_mask.shape}"
        )
        key_padding_mask = key_padding_mask.view(batch_size, 1, 1, src_len).expand(
            -1, num_heads, -1, -1
        ).reshape(batch_size * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    if not training:
        dropout_p = 0.0
    attn_output, attn_output_weights = original_scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
    )
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, batch_size, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        attn_output_weights = attn_output_weights.view(batch_size, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads

    return attn_output, None


class OriginalMultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,
                 kdim=None, vdim=None, batch_first=False) -> None:
        super(OriginalMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == self.embed_dim and self.vdim == self.embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim)))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim)))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim)))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim)))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True,
                attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        attn_output, attn_output_weights = original_multi_head_attention_forward(
            query, key, value, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights

        return attn_output, attn_output_weights


class OriginalTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False) -> None:
        super(OriginalTransformerEncoderLayer, self).__init__()
        self.self_attn = OriginalMultiheadAttention(d_model, nhead, dropout=dropout,
                                                    batch_first=batch_first)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


class OriginalTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False) -> None:
        super(OriginalTransformerDecoderLayer, self).__init__()
        self.self_attn = OriginalMultiheadAttention(d_model, nhead, dropout=dropout,
                                                    batch_first=batch_first)
        self.multihead_attn = OriginalMultiheadAttention(d_model, nhead, dropout=dropout,
                                                         batch_first=batch_first)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class SharedTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(SharedTransformerEncoder, self).__init__()
        self.layer = encoder_layer
        self.num_layers = num_layers
        self.norm = norm

    @property
    def layers(self):
        return [self.layer for _ in range(self.num_layers)]

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for _ in range(self.num_layers):
            output = self.layer(output, src_mask=mask,
                                src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class SharedTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(SharedTransformerDecoder, self).__init__()
        self.layer = decoder_layer
        self.num_layers = num_layers
        self.norm = norm

    @property
    def layers(self):
        return [self.layer for _ in range(self.num_layers)]

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt
        for _ in range(self.num_layers):
            output = self.layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class IndependentTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(IndependentTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class IndependentTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(IndependentTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


def build_transformer_stack(d_model, nhead, num_encoder_layers, num_decoder_layers,
                            dim_feedforward, dropout, activation, layer_norm_eps,
                            batch_first, shared_layers, transformer_backend):
    if transformer_backend == "torch":
        layer_args = {
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "activation": "relu" if activation is F.relu else activation,
            "layer_norm_eps": layer_norm_eps,
            "batch_first": batch_first,
        }
        encoder_layer = nn.TransformerEncoderLayer(**layer_args)
        decoder_layer = nn.TransformerDecoderLayer(**layer_args)
        independent_encoder = nn.TransformerEncoder
        independent_decoder = nn.TransformerDecoder
    elif transformer_backend == "original":
        layer_args = {
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "activation": activation,
            "layer_norm_eps": layer_norm_eps,
            "batch_first": batch_first,
        }
        encoder_layer = OriginalTransformerEncoderLayer(**layer_args)
        decoder_layer = OriginalTransformerDecoderLayer(**layer_args)
        independent_encoder = IndependentTransformerEncoder
        independent_decoder = IndependentTransformerDecoder
    else:
        raise ValueError(
            f"Unknown transformer_backend '{transformer_backend}'. "
            "Valid choices: ['torch', 'original']"
        )

    encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
    decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
    if shared_layers:
        encoder = SharedTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        decoder = SharedTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
    else:
        encoder = independent_encoder(encoder_layer, num_encoder_layers, encoder_norm)
        decoder = independent_decoder(decoder_layer, num_decoder_layers, decoder_norm)

    return encoder, decoder


class Transformer(nn.Module):

    def __init__(self, d_model: int = 64,
                 custom_encoder: Optional[Any] = None,
                 custom_decoder: Optional[Any] = None,
                 reduction=64, channel: int = 2, nt: int = 32,
                 nc: int = 32, shared_layers: bool = True,
                 transformer_backend: str = "torch") -> None:
        super(Transformer, self).__init__()
        if custom_encoder is None or custom_decoder is None:
            raise ValueError("custom_encoder and custom_decoder must be provided")
        self.encoder = custom_encoder
        self.decoder = custom_decoder

        self.channel = channel
        self.nt = nt
        self.nc = nc
        self.input_dim = channel * nt * nc
        self.d_model = d_model

        assert self.input_dim % self.d_model == 0, (
            f"d_model needs to divide the flattened CSI size ({self.input_dim})")
        assert self.input_dim % reduction == 0, (
            f"reduction needs to divide the flattened CSI size ({self.input_dim})")
        self.feature_shape = (self.input_dim // self.d_model, self.d_model)

        self.shared_layers = shared_layers
        self.transformer_backend = transformer_backend
        self.fc_encoder = nn.Linear(self.input_dim, self.input_dim // reduction)
        self.fc_decoder = nn.Linear(self.input_dim // reduction, self.input_dim)
        self._reset_parameters()

    def forward(self, src: Tensor, tgt: Optional[Tensor] = None,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src_shape = src.shape
        src = src.view(-1, self.feature_shape[0], self.feature_shape[1])
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        memory_encoder = self.fc_encoder(memory.view(memory.shape[0], -1))
        memory_decoder = self.fc_decoder(memory_encoder).view(-1, self.feature_shape[0], self.feature_shape[1])
        output = self.decoder(
            memory_decoder,
            memory_decoder,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        output = output.reshape(src_shape)
        return output

    # def encode(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
    #     batch_size = src.size(0)
    #     src = src.view(src.size(0), self.feature_shape[0], self.feature_shape[1])
    #     memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    #     memory_encoder = self.fc_encoder(memory.reshape(batch_size, -1))
    #     return memory_encoder

    def encode(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src = src.view(-1, self.feature_shape[0], self.feature_shape[1])
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        memory_encoder = self.fc_encoder(memory.view(memory.shape[0], -1))
        return memory_encoder

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


def transnet(reduction=64, d_model=64, channel=2, nt=32, nc=32, dim_feedforward=None, shared_layers=True, transformer_backend="torch"):
    r""" Create a proposed TransNet.

        :param reduction: the reciprocal of compression ratio
        :return: an instance of TransNet
    """
    dim_feedforward = 4 * d_model if dim_feedforward is None else dim_feedforward

    encoder, decoder = build_transformer_stack(
        d_model=d_model,
        nhead=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=dim_feedforward,
        dropout=0.,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=True,
        shared_layers=shared_layers,
        transformer_backend=transformer_backend,
    )
    model = Transformer(d_model=d_model,
                        custom_encoder=encoder,
                        custom_decoder=decoder,
                        reduction=reduction,
                        channel=channel,
                        nt=nt,
                        nc=nc,
                        shared_layers=shared_layers,
                        transformer_backend=transformer_backend)
    return model
