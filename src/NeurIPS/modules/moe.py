from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    
    def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,), weight=self.weight, eps=self.eps)

class MLP(nn.Module):
    
    def __init__(self, dim: int, inter_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
class ConditionalGate(nn.Module):
    
    def __init__(
        self,
        dim: int,
        n_routed_experts: int,
        n_activated_experts: int,
        score_func: Literal["softmax", "sigmoid"] = "softmax",
        route_scale: float = 1.0
    ) -> None:
        super().__init__()
        
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.topk = n_activated_experts
        self.score_func = score_func
        self.route_scale = route_scale
        
        self.scores_w = nn.Linear(dim, n_routed_experts, bias=False)
        self.scores_cond_w = nn.Linear(dim, n_routed_experts, bias=False)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.IntTensor]:
        scores = self.scores_w(x) + self.scores_cond_w(cond)
        
        if self.score_func == "softmax":
            scores = torch.softmax(scores, dim=-1, dtype=torch.float32)
        elif self.score_func == "sigmoid":
            scores = torch.sigmoid(scores)
        else:
            raise ValueError(f"unknown score_func '{self.score_func}'")
        
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = torch.gather(scores, dim=1, index=indices)
        
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        
        return weights.type_as(x), indices

class ConditionalMoE(nn.Module):
    
    def __init__(
        self,
        dim: int,
        inter_dim: int,
        n_routed_experts: int,
        n_activated_experts: int,
        n_shared_experts: int,
        score_func: Literal["softmax", "sigmoid"] = "softmax",
        route_scale: float = 1.0
    ) -> None:
        super().__init__()
        
        self.dim = dim
        self.inter_dim = inter_dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.n_shared_experts = n_shared_experts
        self.score_func = score_func
        self.route_scale = route_scale
        
        self.gate = ConditionalGate(
            dim=dim,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            score_func=score_func,
            route_scale=route_scale
        )
        
        self.experts = nn.ModuleList([MLP(dim, inter_dim) for _ in range(self.n_routed_experts)])
        self.shared_experts = MLP(dim, n_shared_experts * inter_dim)
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:

        shape = x.size()
        x = x.view(-1, self.dim)
        cond = cond.unsqueeze(1).expand(-1, shape[1], -1)
        cond = cond.reshape(-1, self.dim)
        
        # route
        weights, indices = self.gate(x, cond)
        y = torch.zeros_like(x)
        
        # routed experts
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.n_routed_experts):
            if counts[i] == 0: continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        
        # shared experts
        z = self.shared_experts(x)
        
        return (y + z).view(shape)

class SelfAttention(nn.Module):

    def __init__(
        self, 
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        causal_attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output

class ConditionalMoEBlock(nn.Module):
    
    def __init__(
        self,
        dim: int = 768,
        n_heads: int = 12,
        dropout: float = 0.0,
        inter_dim: int = 3072,
        moe_inter_dim: int = 512,
        n_routed_experts: int = 64,
        n_activated_experts: int = 6,
        n_shared_experts: int = 2,
        score_func: Literal["softmax", "sigmoid"] = "softmax",
        route_scale: float = 1.0
    ) -> None:
        super().__init__()
        
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.inter_dim = inter_dim
        
        self.self_attn = SelfAttention(dim, n_heads, dropout)
        self.norm1 = RMSNorm(dim=dim)        
        self.moe = ConditionalMoE(
            dim=dim,
            inter_dim=moe_inter_dim,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            n_shared_experts=n_shared_experts,
            score_func=score_func,
            route_scale=route_scale
        )
        self.norm2 = RMSNorm(dim=dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond: torch.Tensor,
        attention_mask: torch.Tensor = None,
        causal_attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.moe(hidden_states, cond)
        hidden_states = residual + hidden_states
        
        return hidden_states