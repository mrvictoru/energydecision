"""
Modernized Decision Transformer with Pre-Norm, RMSNorm, SwiGLU, PyTorch SDPA, and optional RoPE.
Preserves the original API while improving speed and stability.
"""

# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
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
    def __init__(self, h_dim, max_T, n_heads, drop_p, rope_enabled=False, rope_max_position=4096, rope_base=10000.0):
        super().__init__()
        assert h_dim % n_heads == 0, "h_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = h_dim // n_heads
        self.drop_p = drop_p
        self.rope_enabled = rope_enabled
        if rope_enabled:
            if self.head_dim % 2 != 0:
                raise ValueError("RoPE requires even head_dim")
            self.rotary = RotaryEmbedding(self.head_dim, max_position=rope_max_position, base=rope_base)

        self.qkv = nn.Linear(h_dim, 3 * h_dim, bias=False)
        self.proj = nn.Linear(h_dim, h_dim)
        self.proj_drop = nn.Dropout(drop_p)

        mask = torch.tril(torch.ones(max_T, max_T)).view(1, 1, max_T, max_T)
        self.register_buffer('mask', mask)

    def forward(self, x, key_padding_mask=None):
        # x: [B, T, H]
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, N, T, D]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.rope_enabled:
            cos, sin = self.rotary(T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

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
    def __init__(self, h_dim, max_T, n_heads, drop_p, rope_enabled=False, rope_max_position=4096, rope_base=10000.0):
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
        )
        self.norm2 = RMSNorm(h_dim)
        self.ffn = SwiGLU(h_dim, 4 * h_dim, drop_p)

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.ffn(self.norm2(x))
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
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.context_len = context_len

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
            )
            for _ in range(n_block)
        ]
        self.transformer = nn.ModuleList(blocks)

        # projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = nn.Linear(1, h_dim)
        self.embed_state = nn.Linear(state_dim, h_dim)

        # discrete actions
        #self.embed_act = torch.nn.Embedding(act_dim+1, h_dim)
        #use_action_tah = False # for discrete action

        # continuous actions
        self.embed_act = nn.Linear(act_dim, h_dim)
        use_action_tah = True # for continuous action

        # prediction heads
        self.pred_rtg = nn.Linear(h_dim, 1)
        self.pred_state = nn.Linear(h_dim, state_dim)
        self.pred_act = nn.Sequential(*([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tah else [])))
        # final norm
        self.ln_f = RMSNorm(h_dim)
        
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
        return_preds = self.pred_rtg(h[:,2])    # predict next rtg given r, s, a
        state_preds = self.pred_state(h[:,2])   # predict next state given r, s, a
        act_preds = self.pred_act(h[:,1])       # predict action given r, s

        return_preds = torch.nan_to_num(return_preds, nan=0.0, posinf=0.0, neginf=0.0)
        state_preds = torch.nan_to_num(state_preds, nan=0.0, posinf=0.0, neginf=0.0)
        act_preds = torch.nan_to_num(act_preds, nan=0.0, posinf=0.0, neginf=0.0)

        return return_preds, state_preds, act_preds
    
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