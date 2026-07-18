"""
Modernized Decision Transformer with Pre-Norm, RMSNorm, SwiGLU, PyTorch SDPA, and optional RoPE.
Preserves the original API while improving speed and stability.
"""

# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x * x, dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm_x + self.eps)
        return x * self.scale


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, drop_p):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position=4096, base=10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE requires even dimension")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("cos", torch.cos(freqs))
        self.register_buffer("sin", torch.sin(freqs))

    def forward(self, seq_len):
        if seq_len > self.cos.shape[0]:
            raise ValueError("seq_len exceeds RoPE cache")
        return self.cos[:seq_len], self.sin[:seq_len]


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.to(dtype=q.dtype, device=q.device).unsqueeze(0).unsqueeze(0)
    sin = sin.to(dtype=q.dtype, device=q.device).unsqueeze(0).unsqueeze(0)
    q_even, q_odd = q[..., ::2], q[..., 1::2]
    k_even, k_odd = k[..., ::2], k[..., 1::2]
    q = torch.cat([q_even * cos - q_odd * sin, q_even * sin + q_odd * cos], dim=-1)
    k = torch.cat([k_even * cos - k_odd * sin, k_even * sin + k_odd * cos], dim=-1)
    return q, k


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        h_dim,
        max_T,
        n_heads,
        drop_p,
        rope_enabled=False,
        rope_max_position=4096,
        rope_base=10000.0,
        n_kv_heads=None,
        qk_norm=False,
    ):
        super().__init__()
        assert h_dim % n_heads == 0, "h_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = h_dim // n_heads
        self.drop_p = drop_p
        self.rope_enabled = rope_enabled

        # Grouped Query Attention: n_kv_heads <= n_heads (default: full MHA)
        if n_kv_heads is None:
            n_kv_heads = n_heads
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # how many times to repeat K/V

        if rope_enabled:
            if self.head_dim % 2 != 0:
                raise ValueError("RoPE requires even head_dim")
            self.rotary = RotaryEmbedding(self.head_dim, max_position=rope_max_position, base=rope_base)

        # Separate Q, K, V projections (bias-free, Llama-style)
        self.q_proj = nn.Linear(h_dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h_dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h_dim, n_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(h_dim, h_dim, bias=False)
        self.proj_drop = nn.Dropout(drop_p)

        # Optional QK-Norm for training stability at larger scales
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        mask = torch.tril(torch.ones(max_T, max_T)).view(1, 1, max_T, max_T)
        self.register_buffer('mask', mask)

    def forward(self, x, key_padding_mask=None):
        # x: [B, T, H]
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)     # [B, n_heads, T, D]
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, n_kv_heads, T, D]
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, n_kv_heads, T, D]

        # QK-Norm (applied per head, before RoPE)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.rope_enabled:
            cos, sin = self.rotary(T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand K/V to match n_heads (GQA repeat)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)  # [B, n_heads, T, D]
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Build combined attention mask
        causal = self.mask[:, :, :T, :T].bool()  # [1,1,T,T]

        if key_padding_mask is not None:
            kp = key_padding_mask.view(B, 1, 1, T).to(dtype=torch.bool)
            combined = causal & kp
            # Float mask: 0 keep, -inf mask
            attn_mask = torch.zeros((B, 1, T, T), device=x.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~combined, float('-inf'))
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.drop_p if self.training else 0.0
            )
        else:
            y = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=self.drop_p if self.training else 0.0
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_drop(self.proj(y))
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return y


class ModernBlock(nn.Module):
    def __init__(
        self,
        h_dim,
        max_T,
        n_heads,
        drop_p,
        rope_enabled=False,
        rope_max_position=4096,
        rope_base=10000.0,
        n_kv_heads=None,
        qk_norm=False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(h_dim)
        self.attn = CausalSelfAttention(
            h_dim,
            max_T,
            n_heads,
            drop_p,
            rope_enabled=rope_enabled,
            rope_max_position=rope_max_position,
            rope_base=rope_base,
            n_kv_heads=n_kv_heads,
            qk_norm=qk_norm,
        )
        self.norm2 = RMSNorm(h_dim)
        self.ffn = SwiGLU(h_dim, 4 * h_dim, drop_p)

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class LegacyRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class LegacySwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class LegacyDecisionTransformer(nn.Module):
    """Compatibility model for older checkpoint exports that use the MoLab-style architecture."""

    def __init__(
        self,
        state_dim,
        act_dim,
        n_block=8,
        h_dim=384,
        context_len=180,
        n_heads=8,
        drop_p=0.1,
        max_timestep=100000,
        use_rope=False,
        rope_base=10000.0,
        rope_max_position=None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.context_len = context_len
        self.h_dim = h_dim
        self.n_heads = n_heads
        self.drop_p = drop_p
        self.max_timestep = max_timestep
        self.use_rope = use_rope

        self.embed_return = nn.Linear(1, h_dim)
        self.embed_state = nn.Linear(state_dim, h_dim)
        self.embed_action = nn.Linear(act_dim, h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)

        self.blocks = nn.ModuleList(
            [
                LegacyTransformerBlock(h_dim, n_heads, drop_p, use_rope, rope_base, rope_max_position)
                for _ in range(n_block)
            ]
        )
        self.ln_f = LegacyRMSNorm(h_dim)

        self.predict_return = nn.Linear(h_dim, 1)
        self.predict_state = nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, act_dim),
            nn.Tanh(),
        )

        self.return_scale = nn.Parameter(torch.tensor(2.0), requires_grad=False)

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        B, T, _ = states.shape
        if returns_to_go.dim() == 2:
            returns_to_go = returns_to_go.unsqueeze(-1)
        elif returns_to_go.dim() == 1:
            returns_to_go = returns_to_go.unsqueeze(0).unsqueeze(-1)
        elif returns_to_go.dim() == 3 and returns_to_go.shape[-1] != 1:
            returns_to_go = returns_to_go.unsqueeze(-1)

        time_emb = self.embed_timestep(timesteps)
        state_emb = self.embed_state(states)
        action_emb = self.embed_action(actions)
        return_emb = self.embed_return(returns_to_go)

        stacked = torch.stack([return_emb, state_emb, action_emb], dim=2)
        x = stacked.permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)
        x = x + time_emb.repeat_interleave(3, dim=1)

        if attention_mask is not None:
            attn_mask = attention_mask.repeat_interleave(3, dim=1)
            attn_mask = attn_mask.unsqueeze(1)
        else:
            attn_mask = None

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)

        pred_mask = torch.zeros(3 * T, dtype=torch.bool)
        pred_mask[0::3] = True
        pred_mask[1::3] = True
        act_mask = torch.ones(3 * T, dtype=torch.bool)
        act_mask[2::3] = False

        x_pred = x[:, pred_mask]
        x_act = x[:, act_mask]

        return_preds = self.predict_return(x_pred[:, ::2])
        state_preds = self.predict_state(x_pred[:, 1::2])
        action_preds = self.predict_action(x_act[:, ::2])
        return action_preds, state_preds, return_preds

    def get_action(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        action_preds, _, _ = self.forward(states, actions, returns_to_go, timesteps, attention_mask)
        return action_preds[:, -1]


class LegacyTransformerBlock(nn.Module):
    def __init__(self, h_dim, n_heads, drop_p=0.1, use_rope=False, rope_base=10000.0, rope_max_position=None):
        super().__init__()
        self.ln1 = LegacyRMSNorm(h_dim)
        self.attn = nn.MultiheadAttention(h_dim, n_heads, dropout=drop_p, batch_first=True)
        self.ln2 = LegacyRMSNorm(h_dim)
        self.ffn = LegacySwiGLU(h_dim)
        self.dropout = nn.Dropout(drop_p)
        self.use_rope = use_rope

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(h)
        h = self.ln2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)
        return x


# define the decision transformer
class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim,
        act_dim,
        n_block,
        h_dim,
        context_len,
        n_heads,
        drop_p,
        max_timestep=4096,
        rope_enabled=False,
        rope_max_position=4096,
        rope_base=10000.0,
        n_kv_heads=None,
        qk_norm=False,
        tie_weights=False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.context_len = context_len
        self.n_heads = n_heads
        self.n_block = n_block
        self.drop_p = drop_p
        self.max_timestep = max_timestep
        self.rope_enabled = rope_enabled

        # transformer blocks
        input_seq_len = 3 * context_len
        blocks = [
            ModernBlock(
                h_dim,
                input_seq_len,
                n_heads,
                drop_p,
                rope_enabled=rope_enabled,
                rope_max_position=rope_max_position,
                rope_base=rope_base,
                n_kv_heads=n_kv_heads,
                qk_norm=qk_norm,
            )
            for _ in range(n_block)
        ]
        self.transformer = nn.ModuleList(blocks)

        # projection heads (project to embedding)
        self.embed_ln = RMSNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = nn.Linear(1, h_dim)
        self.embed_state = nn.Linear(state_dim, h_dim)

        # continuous actions
        self.embed_act = nn.Linear(act_dim, h_dim)

        # final norm
        self.ln_f = RMSNorm(h_dim)

        # prediction heads (bias-free) + optional weight tying
        # When tie_weights=True: pred_* modules have swapped in/out dims so their
        # weight tensors have the SAME shape as the embed_* weights.  We then assign
        # pred_*.weight = embed_*.weight (same nn.Parameter object) so PyTorch's
        # parameter deduplication eliminates the redundant storage.  The forward
        # method uses F.linear(h, weight.t()) to produce the correct output shape.
        self._tie_weights = tie_weights
        if tie_weights:
            # Swapped dims: Linear(out_orig, h_dim) so weight.shape == embed_*.weight.shape
            self.pred_rtg = nn.Linear(1, h_dim, bias=False)           # weight: (h_dim, 1)
            self.pred_state = nn.Linear(state_dim, h_dim, bias=False)  # weight: (h_dim, state_dim)
            _pred_act_linear = nn.Linear(act_dim, h_dim, bias=False)  # weight: (h_dim, act_dim)
            # True parameter sharing: same nn.Parameter object as embed_*
            self.pred_rtg.weight = self.embed_rtg.weight
            self.pred_state.weight = self.embed_state.weight
            _pred_act_linear.weight = self.embed_act.weight
            self.pred_act = nn.Sequential(_pred_act_linear, nn.Tanh())
        else:
            self.pred_rtg = nn.Linear(h_dim, 1, bias=False)
            self.pred_state = nn.Linear(h_dim, state_dim, bias=False)
            self.pred_act = nn.Sequential(nn.Linear(h_dim, act_dim, bias=False), nn.Tanh())

        # Default return_scale for inference (can be set during training)
        self.return_scale = 1.0
    
    def forward(self, state, rtg, timestep, actions, attention_mask=None):
        B, T, _ = state.shape

        # Sanitize inputs to avoid propagating NaNs/Infs
        state = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        rtg = torch.nan_to_num(rtg, nan=0.0, posinf=0.0, neginf=0.0)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)

        # Clamp timestep indices to embedding range
        max_timestep = self.embed_timestep.num_embeddings
        timestep = timestep.clamp(min=0, max=max_timestep - 1)

        # timestep embedding
        time_emb = self.embed_timestep(timestep)

        # embedding for the state, reward and actions along with time embedding
        state_emb = self.embed_state(state) + time_emb
        rtg_emb = self.embed_rtg(rtg) + time_emb
        act_emb = self.embed_act(actions) + time_emb

        # stack the embeddings and reshape sequence as (r1, s1, a1, r2, s2, a2, ...)
        h = torch.stack([rtg_emb, state_emb, act_emb], dim=1).permute(0,2,1,3).reshape(B, 3*T, self.h_dim)
        h = self.embed_ln(h)

        # Build stacked attention mask for the tripled sequence (R, s, a)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=h.device)
            # Convert to boolean mask (True = valid token)
            attention_mask = attention_mask > 0
            # Stack it three times for (rtg, state, action) tokens
            stacked_mask = torch.stack([attention_mask, attention_mask, attention_mask], dim=1)  # [B, 3, T]
            stacked_mask = stacked_mask.permute(0, 2, 1).reshape(B, 3*T)  # [B, 3*T]
        else:
            # Default to all ones (all tokens are valid)
            stacked_mask = torch.ones(B, 3*T, dtype=torch.bool, device=h.device)

        # transformer blocks
        for block in self.transformer:
            h = block(h, key_padding_mask=stacked_mask)

        # final pre-norm on output stream
        h = self.ln_f(h)

        # get h reshaped such that its size is (B, 3, T, h_dim) and
        # h[:, 0, t] is conditioned on r_0, s_0, a_0, ..., r_t
        # h[:, 1, t] is conditioned on r_0, s_0, a_0, ..., r_t, s_t
        # h[:, 2, t] is conditioned on r_0, s_0, a_0, ..., r_t, s_t, a_t
        h = h.reshape(B, T, 3, self.h_dim).permute(0,2,1,3)

        # get predictions
        if self._tie_weights:
            # Tied mode: weight shape is (h_dim, out_dim); use F.linear(h, W.t()) = h @ W
            return_preds = F.linear(h[:,2], self.pred_rtg.weight.t())    # (B,T,1)
            state_preds = F.linear(h[:,2], self.pred_state.weight.t())   # (B,T,state_dim)
            act_preds = torch.tanh(F.linear(h[:,1], self.pred_act[0].weight.t()))  # (B,T,act_dim)
        else:
            return_preds = self.pred_rtg(h[:,2])    # predict next rtg given r, s, a
            state_preds = self.pred_state(h[:,2])   # predict next state given r, s, a
            act_preds = self.pred_act(h[:,1])       # predict action given r, s

        return_preds = torch.nan_to_num(return_preds, nan=0.0, posinf=0.0, neginf=0.0)
        state_preds = torch.nan_to_num(state_preds, nan=0.0, posinf=0.0, neginf=0.0)
        act_preds = torch.nan_to_num(act_preds, nan=0.0, posinf=0.0, neginf=0.0)

        return return_preds, state_preds, act_preds
    
    def load_from_checkpoint(self, checkpoint_or_state, map_location=None, strict: bool = True):
        """
        Robust loading of a checkpoint (path or state-dict). If the checkpoint contains
        RoPE buffers (rotary.cos / rotary.sin) this will create RotaryEmbedding
        instances with matching max_position for each attention block before loading.
        Falls back to strict=False if strict=True fails.
        """
        # load from disk if a path was provided
        sidecar_meta = None
        if isinstance(checkpoint_or_state, (str, bytes)):
            ckpt_path = checkpoint_or_state.decode() if isinstance(checkpoint_or_state, (bytes, bytearray)) else str(checkpoint_or_state)
            state = torch.load(ckpt_path, map_location=map_location)

            # Best-effort: load inference metadata from sidecar
            meta_path = ckpt_path + ".meta.json"
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        sidecar_meta = json.load(f)
                except Exception:
                    sidecar_meta = None
        else:
            state = checkpoint_or_state

        # Accept multiple formats:
        # 1) raw state_dict (backwards compatible)
        # 2) training checkpoint dict from transformer_training.py
        # 3) a bundle dict with keys like {model_state_dict, meta}
        meta = None
        if isinstance(state, dict) and "model_state_dict" in state and isinstance(state.get("model_state_dict"), dict):
            meta = state.get("meta") if isinstance(state.get("meta"), dict) else None
            # transformer_training checkpoint stores return_scale at top-level
            if meta is None and "return_scale" in state:
                meta = {"return_scale": state.get("return_scale")}
            state = state["model_state_dict"]

        if meta is None and isinstance(sidecar_meta, dict):
            meta = sidecar_meta

        if isinstance(meta, dict) and "return_scale" in meta:
            try:
                rs = float(meta["return_scale"])
                if rs == rs and abs(rs) >= 1e-12:  # finite + non-zero
                    self.return_scale = rs
            except Exception:
                pass

        # Detect legacy (MoLab-style) checkpoints by keys that are exclusive to them.
        # "embed_state" appears in BOTH legacy and modern models so it must NOT be used here.
        # Discriminating keys: embed_return (modern uses embed_rtg), embed_action (modern uses
        # embed_act), predict_* (modern uses pred_*), and blocks.* (modern uses transformer.*).
        legacy_checkpoint = any(
            key.startswith("embed_return")
            or key.startswith("embed_action")
            or key.startswith("predict_return")
            or key.startswith("predict_state")
            or key.startswith("predict_action")
            or key.startswith("blocks.")
            for key in state.keys()
        )

        if legacy_checkpoint:
            legacy_state_dim = self.state_dim
            legacy_act_dim = self.act_dim
            legacy_h_dim = self.h_dim
            legacy_max_timestep = self.max_timestep

            embed_state = state.get("embed_state.weight")
            if isinstance(embed_state, torch.Tensor) and embed_state.ndim == 2:
                legacy_state_dim = int(embed_state.shape[1])
                legacy_h_dim = int(embed_state.shape[0])

            embed_action = state.get("embed_action.weight")
            if isinstance(embed_action, torch.Tensor) and embed_action.ndim == 2:
                legacy_act_dim = int(embed_action.shape[1])
                legacy_h_dim = int(embed_action.shape[0])

            embed_timestep = state.get("embed_timestep.weight")
            if isinstance(embed_timestep, torch.Tensor) and embed_timestep.ndim == 2:
                legacy_max_timestep = int(embed_timestep.shape[0])
                legacy_h_dim = int(embed_timestep.shape[1])

            legacy_model = LegacyDecisionTransformer(
                state_dim=legacy_state_dim,
                act_dim=legacy_act_dim,
                n_block=self.n_block,
                h_dim=legacy_h_dim,
                context_len=self.context_len,
                n_heads=self.n_heads,
                drop_p=self.drop_p,
                max_timestep=legacy_max_timestep,
                use_rope=False,
            )
            if isinstance(meta, dict) and "return_scale" in meta:
                try:
                    legacy_model.return_scale.data.copy_(torch.tensor(float(meta["return_scale"])))
                except Exception:
                    pass
            legacy_model.load_state_dict(state, strict=False)
            self.__class__ = LegacyDecisionTransformer
            self.__dict__.update(legacy_model.__dict__)
            return

        # detect rotary cos buffers and patch local blocks to match checkpoint shape
        cos_keys = [k for k in state.keys() if k.endswith(".rotary.cos")]
        if cos_keys:
            for k in cos_keys:
                # expected format: "transformer.{i}.attn.rotary.cos"
                parts = k.split(".")
                idx = next((int(p) for p in parts if p.isdigit()), None)
                if idx is None or not hasattr(self, "transformer") or idx >= len(self.transformer):
                    continue
                cos_buf = state[k]
                max_pos = int(cos_buf.shape[0])
                half_dim = int(cos_buf.shape[1])
                inferred_dim = half_dim * 2

                block = self.transformer[idx]
                attn = getattr(block, "attn", None)
                if attn is None:
                    continue

                head_dim = getattr(attn, "head_dim", None)
                if head_dim is None:
                    head_dim = inferred_dim
                # ensure even head dim
                if head_dim % 2 != 0:
                    head_dim = inferred_dim

                # create a RotaryEmbedding with matching max_position
                attn.rotary = RotaryEmbedding(head_dim, max_position=max_pos)
                attn.rope_enabled = True

        # Migrate old fused-QKV checkpoints (pre-GQA) to the new split-projection format.
        # Old key: "transformer.{i}.attn.qkv.weight"  shape [3*h_dim, h_dim]
        # New keys: q_proj / k_proj / v_proj            each [h_dim, h_dim]
        qkv_keys = [k for k in list(state.keys()) if k.endswith(".attn.qkv.weight")]
        if qkv_keys:
            state = dict(state)  # copy so we can mutate
            for qkv_key in qkv_keys:
                prefix = qkv_key[: -len("qkv.weight")]  # e.g. "transformer.0.attn."
                w = state.pop(qkv_key)                    # [3*h_dim, h_dim]
                h = w.shape[0] // 3
                state[prefix + "q_proj.weight"] = w[:h]
                state[prefix + "k_proj.weight"] = w[h : 2 * h]
                state[prefix + "v_proj.weight"] = w[2 * h :]

        # attempt strict load, fall back to non-strict if needed
        try:
            self.load_state_dict(state, strict=strict)
        except RuntimeError as e:
            if strict:
                print("Strict load failed, retrying with strict=False (missing/unexpected keys may be present).")
                self.load_state_dict(state, strict=False)
            else:
                raise e
    
    def get_action(self, states, actions, rtg, timesteps, attention_mask=None):
        """
        Convenience method to get the last action prediction from the model.
        Provides API parity with the official Decision Transformer implementation.
        
        Args:
            states: Tensor of shape [B, T, state_dim] or [T, state_dim]
            actions: Tensor of shape [B, T, act_dim] or [T, act_dim]
            rtg: Tensor of shape [B, T, 1] or [T, 1] or [B, T] or [T]
            timesteps: Tensor of shape [B, T] or [T]
            attention_mask: Optional tensor of shape [B, T] or [T] with 1 for valid, 0 for padding
            
        Returns:
            Tensor of shape [act_dim] representing the action prediction at the last position
        """
        # Ensure all inputs have batch dimension
        if states.dim() == 2:  # [T, state_dim]
            states = states.unsqueeze(0)  # [1, T, state_dim]
        if actions.dim() == 2:  # [T, act_dim]
            actions = actions.unsqueeze(0)  # [1, T, act_dim]
        if rtg.dim() == 1:  # [T]
            rtg = rtg.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        elif rtg.dim() == 2:  # Could be [B, T] or [T, 1]
            if rtg.shape[1] == 1 or rtg.shape[0] == 1:  # [T, 1] case
                if rtg.shape[0] != 1:
                    rtg = rtg.unsqueeze(0)  # [1, T, 1]
            else:  # [B, T] case
                rtg = rtg.unsqueeze(-1)  # [B, T, 1]
        if timesteps.dim() == 1:  # [T]
            timesteps = timesteps.unsqueeze(0)  # [1, T]
        if attention_mask is not None and attention_mask.dim() == 1:  # [T]
            attention_mask = attention_mask.unsqueeze(0)  # [1, T]

        # Clamp timesteps to the embedding range before forward pass
        max_timestep = self.embed_timestep.num_embeddings
        timesteps = timesteps.clamp(min=0, max=max_timestep - 1)

        # Sanitize tensors before forward pass
        states = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        rtg = torch.nan_to_num(rtg, nan=0.0, posinf=0.0, neginf=0.0)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool, device=states.device)
        
        # Call forward pass
        _, _, act_preds = self.forward(states, rtg, timesteps, actions, attention_mask=attention_mask)
        act_preds = torch.nan_to_num(act_preds, nan=0.0, posinf=0.0, neginf=0.0)

        # Return the last action prediction, removing batch dimension
        return act_preds[0, -1]