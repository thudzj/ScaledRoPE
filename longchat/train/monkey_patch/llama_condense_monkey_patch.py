# code adapted from https://huggingface.co/kaiokendev/superhot-13b-8k-no-rlhf-test/blob/main/llama_rope_scaled_monkey_patch.py
import torch, math
import transformers
import transformers.models.llama.modeling_llama
from einops import rearrange

from functools import partial
from ..utils import rank0_print

class CondenseRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, ratio, max_position_embeddings=2048, base=10000, device=None, interpolation_type='linear'):
        super().__init__()
        self.interpolation_type = interpolation_type
        self.max_position_embeddings = max_position_embeddings
        self.dim = dim
        self.base = base
        self.device = device
        self.set_ratio(ratio, self.device)
    
    def set_ratio(self, ratio, device):
        if self.interpolation_type == 'ntk' or self.interpolation_type == 'our':
            base = self.base * ratio ** (self.dim / (self.dim-2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        else:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
        max_position_embeddings = int(self.max_position_embeddings * ratio)
        # rank0_print(f"Condensing Positional embeddings from {max_position_embeddings} to {int(max_position_embeddings / ratio)}")
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        if self.interpolation_type == 'linear':
            t /= ratio
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            if self.interpolation_type == 'linear':
                t /= self.ratio
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def replace_llama_with_condense(ratio, interpolation_type='linear'):
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = partial(CondenseRotaryEmbedding, ratio=ratio, interpolation_type=interpolation_type)
