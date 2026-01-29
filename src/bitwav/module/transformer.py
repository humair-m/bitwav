"""
Transformer module with RoPE, local attention, and AdaLNZero conditioning.
Adapted from: https://github.com/meta-llama/llama3/blob/main/llama/model.py
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple, List, Union

from ..util import get_logger
from .adaln_zero import AdaLNZero

logger = get_logger()

# Check for FlashAttention availability
try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    logger.warning(
        "FlashAttention is not installed. Falling back to PyTorch SDPA. "
        "Performance may be reduced and exact match with training setup is not guaranteed."
    )


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency cis for rotary positional embeddings.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape precomputed frequencies for broadcasting with input tensors.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to the input tensor.
    """
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x_)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)


class Attention(nn.Module):
    """
    Multi-head attention module with support for causal, local, and FlashAttention.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        dropout: float,
        window_size: Optional[int],
        qkv_bias: bool = False,
        proj_bias: bool = False,
        use_flash_attention: bool = False,
        causal: bool = False,
    ):
        """
        Initializes the Attention module.

        Args:
            dim (int): Input dimension.
            n_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            window_size (Optional[int]): Window size for local attention.
            qkv_bias (bool): Whether to use bias in QKV projections.
            proj_bias (bool): Whether to use bias in output projection.
            use_flash_attention (bool): Whether to attempt using FlashAttention.
            causal (bool): Whether to apply causal masking.
        """
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, n_heads * self.head_dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, n_heads * self.head_dim, bias=qkv_bias)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=proj_bias)

        self.scale = self.head_dim**-0.5
        self.dropout = dropout

        self.use_local_attention = window_size is not None
        if self.use_local_attention:
            assert window_size % 2 == 1, "Window size must be odd for local attention."
            self.window_per_side = window_size // 2

        self.use_flash_attention = use_flash_attention
        self.causal = causal

    def create_mask(
        self, bsz: int, seqlen: int, mask: Optional[torch.Tensor], device: torch.device
    ) -> Optional[torch.Tensor]:
        """Create attention mask combining causal and local constraints."""
        if not self.use_local_attention and mask is None and not self.causal:
            return None

        attn_mask = torch.ones((seqlen, seqlen), dtype=torch.bool, device=device)

        if self.causal:
            attn_mask = torch.tril(attn_mask)

        if self.use_local_attention:
            attn_mask = torch.triu(attn_mask, diagonal=-self.window_per_side)
            attn_mask = torch.tril(attn_mask, diagonal=self.window_per_side)

        attn_mask = attn_mask.unsqueeze(0).expand(bsz, -1, -1)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).expand(bsz, -1, -1)
            attn_mask = attn_mask & mask

        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        return attn_mask

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        return_kv: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for multi-head attention."""
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        if freqs_cis is not None:
            xq = apply_rotary_emb(xq, freqs_cis=freqs_cis[:seqlen])
            xk = apply_rotary_emb(xk, freqs_cis=freqs_cis[:seqlen])

        if self.use_flash_attention and FLASH_ATTN_AVAILABLE and mask is None:
            window_size = (self.window_per_side, self.window_per_side) if self.use_local_attention else (-1, -1)
            output = flash_attn_func(
                xq, xk, xv,
                dropout_p=(self.dropout if self.training else 0.0),
                softmax_scale=self.scale,
                window_size=window_size,
                causal=self.causal,
            )
        else:
            attn_mask = self.create_mask(bsz, seqlen, mask, x.device)
            output = F.scaled_dot_product_attention(
                xq.transpose(1, 2),
                xk.transpose(1, 2),
                xv.transpose(1, 2),
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale,
            ).transpose(1, 2)

        output = output.contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)

        if return_kv:
            return output, (xk, xv)
        return output

    def forward_with_cache(
        self,
        x: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        freqs_cis: torch.Tensor,
        start_pos: int,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with KV cache for incremental generation."""
        bsz, seqlen, _ = x.shape
        assert seqlen == 1, "KV cache method is for single-token generation."

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis[start_pos : start_pos + seqlen])
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis[start_pos : start_pos + seqlen])

        k_cache, v_cache = kv_cache
        new_kv = (xk, xv)
        xk = torch.cat([k_cache, xk], dim=1)
        xv = torch.cat([v_cache, xv], dim=1)

        if self.use_flash_attention and FLASH_ATTN_AVAILABLE:
            output = flash_attn_with_kvcache(xq, xk, xv, softmax_scale=self.scale)
        else:
            output = F.scaled_dot_product_attention(
                xq.transpose(1, 2),
                xk.transpose(1, 2),
                xv.transpose(1, 2),
                scale=self.scale,
            ).transpose(1, 2)

        output = output.contiguous().view(bsz, seqlen, -1)
        return self.wo(output), new_kv


class FeedForward(nn.Module):
    """
    SwiGLU feed-forward network.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initializes the FeedForward module.
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """
    Standard Transformer block containing attention and feed-forward sub-layers.
    Supports AdaLNZero conditioning.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        qkv_bias: bool,
        proj_bias: bool,
        window_size: Optional[int],
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        dropout: float,
        norm_eps: float,
        adanorm_condition_dim: Optional[int] = None,
        use_flash_attention: bool = False,
        use_adaln_zero: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.attention = Attention(
            dim=dim,
            n_heads=n_heads,
            dropout=dropout,
            window_size=window_size,
            use_flash_attention=use_flash_attention,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            causal=causal,
        )

        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        self.use_adaln_zero = use_adaln_zero
        if self.use_adaln_zero:
            assert adanorm_condition_dim is not None
            self.attention_norm = AdaLNZero(dim, adanorm_condition_dim, eps=norm_eps, return_gate=True)
            self.ffn_norm = AdaLNZero(dim, adanorm_condition_dim, eps=norm_eps, return_gate=True)
        else:
            self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
            self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        condition: Optional[torch.Tensor] = None,
        return_kv: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for the Transformer block."""
        if self.use_adaln_zero:
            attn_normed, attn_gate = self.attention_norm(x, condition=condition)
        else:
            attn_normed = self.attention_norm(x)

        new_kv = None
        if kv_cache is not None and start_pos is not None:
            attn_out, new_kv = self.attention.forward_with_cache(attn_normed, kv_cache, freqs_cis, start_pos)
        elif return_kv:
            attn_out, new_kv = self.attention(attn_normed, freqs_cis, mask, return_kv=True)
        else:
            attn_out = self.attention(attn_normed, freqs_cis, mask)

        if self.use_adaln_zero:
            h = x + attn_gate * attn_out
        else:
            h = x + attn_out

        if self.use_adaln_zero:
            ffn_normed, ffn_gate = self.ffn_norm(h, condition=condition)
        else:
            ffn_normed = self.ffn_norm(h)

        ffn_out = self.feed_forward(ffn_normed)

        if self.use_adaln_zero:
            out = h + ffn_gate * ffn_out
        else:
            out = h + ffn_out

        if new_kv is not None:
            return out, new_kv
        return out


class Transformer(nn.Module):
    """
    Transformer model composed of multiple TransformerBlocks.
    """

    def __init__(
        self,
        dim: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        window_size: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        dropout: float = 0.1,
        norm_eps: float = 1e-5,
        use_rope: bool = True,
        rope_theta: float = 500000.0,
        max_seq_len: int = 2048,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        adanorm_condition_dim: Optional[int] = None,
        use_flash_attention: bool = False,
        use_adaln_zero: bool = False,
        use_xavier_init: bool = True,
        causal: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.rope_theta = rope_theta
        self.use_adaln_zero = use_adaln_zero

        self.layers = nn.ModuleList()
        for layer_id in range(n_layers):
            self.layers.append(
                TransformerBlock(
                    dim=dim, n_heads=n_heads, window_size=window_size,
                    multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier,
                    dropout=dropout, qkv_bias=qkv_bias, proj_bias=proj_bias,
                    norm_eps=norm_eps, adanorm_condition_dim=adanorm_condition_dim,
                    use_flash_attention=use_flash_attention, use_adaln_zero=use_adaln_zero,
                    causal=causal,
                )
            )

        if self.use_adaln_zero:
            self.norm = AdaLNZero(dim, adanorm_condition_dim, eps=norm_eps, return_gate=False)
        else:
            self.norm = nn.LayerNorm(dim, eps=norm_eps)

        self.input_proj = nn.Linear(input_dim, dim) if input_dim is not None else nn.Identity()
        self.output_proj = nn.Linear(dim, output_dim) if output_dim is not None else nn.Identity()
        self.output_dim_ = output_dim if output_dim is not None else dim

        if use_rope:
            self.freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len * 2, rope_theta)
        else:
            self.freqs_cis = None

        if use_xavier_init:
            self.apply(self._init_weights)
            self.apply(self._init_adaln_zero)

    @property
    def output_dim(self) -> int:
        return self.output_dim_

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_adaln_zero(self, module: nn.Module):
        if isinstance(module, AdaLNZero):
            nn.init.zeros_(module.condition_proj[1].weight)
            nn.init.zeros_(module.condition_proj[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        return_kv: bool = False,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass for the Transformer."""
        bsz, seqlen, _dim = x.shape

        if self.freqs_cis is not None:
            expected_len = (start_pos + 1) if start_pos is not None else seqlen
            if expected_len > self.freqs_cis.shape[0]:
                self.freqs_cis = precompute_freqs_cis(self.dim // self.n_heads, expected_len * 4, self.rope_theta)
            self.freqs_cis = self.freqs_cis.to(x.device)
            freqs_cis = self.freqs_cis
        else:
            freqs_cis = None

        x = self.input_proj(x)
        new_kv_list = []
        for i, layer in enumerate(self.layers):
            if kv_cache is not None and start_pos is not None:
                x, new_kv = layer(x, freqs_cis, mask, condition, kv_cache=kv_cache[i], start_pos=start_pos)
                new_kv_list.append(new_kv)
            elif return_kv:
                x, new_kv = layer(x, freqs_cis, mask, condition, return_kv=True)
                new_kv_list.append(new_kv)
            else:
                x = layer(x, freqs_cis, mask, condition)

        if self.use_adaln_zero:
            x, _ = self.norm(x, condition=condition)
        else:
            x = self.norm(x)

        output = self.output_proj(x)

        if new_kv_list:
            return output, new_kv_list
        return output
