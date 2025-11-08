"""
This module contains the classes for creating a custom transformer based model for acting as a decision making agent for the trading environment.
Based on https://github.com/nikhilbarhate99/min-decision-transformer/blob/master/decision_transformer/model.py

"""

# import libraries
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# define the masked causal attention
class MaskedAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.n_heads = n_heads
        self.drop_p = drop_p
        # feed forward networks which create the query, key and value
        self.Q_net = nn.Linear(h_dim, h_dim)
        self.K_net = nn.Linear(h_dim, h_dim)
        self.V_net = nn.Linear(h_dim, h_dim)

        # feed forward network which projects the attention to the correct dimension
        self.proj_net = nn.Linear(h_dim, h_dim)

        # dropout layers
        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        # create the mask
        mask = torch.tril(torch.ones(max_T, max_T)).view(1, 1, max_T, max_T)

        # register_buffer will make the mask a constant tensor
        # so that it will not be included in the model parameters and be updated during backpropagation
        self.register_buffer('mask', mask)

    def forward(self, x, key_padding_mask=None):
        # x: [B, T, H]
        # key_padding_mask: [B, T] with 1 for valid tokens, 0 for padding (optional)
        B, T, C = x.shape # batch size, sequence length, hidden dimension * number of heads
        N, D = self.n_heads, C // self.n_heads # number of heads, dimension of each head

        # compute the query, key and value
        Q = self.Q_net(x).view(B, T, N, D).transpose(1, 2) # [B, N, T, D]
        K = self.K_net(x).view(B, T, N, D).transpose(1, 2)
        V = self.V_net(x).view(B, T, N, D).transpose(1, 2)

        # compute the attention
        weights = Q @ K.transpose(2,3) / math.sqrt(D) # QK^T / sqrt(D)
        
        # Apply causal mask (lower triangular)
        weights = weights.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf')) # mask the future tokens 
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # Convert to boolean mask: True = keep, False = mask out
            key_padding_mask = key_padding_mask.to(dtype=torch.bool, device=weights.device)
            key_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            weights = weights.masked_fill(~key_mask, float('-inf'))
        
        normalized_weights = F.softmax(weights, dim=-1) # softmax along the last dimension
        # Replace NaN/Inf with 0 (happens when all keys are masked for a query position)
        normalized_weights = torch.nan_to_num(normalized_weights, nan=0.0, posinf=0.0, neginf=0.0)
        A = self.att_drop(normalized_weights @ V) # attention with dropout

        # compute the output
        # gather heads and project to correct dimension
        attention = A.transpose(1, 2).contiguous().view(B, T, N*D)
        out = self.proj_drop(self.proj_net(attention))
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        return out

# define the attention block with layer normalization and residual connection as well as the feed forward network
class AttentionBlock(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedAttention(h_dim, max_T, n_heads, drop_p)
        self.norm1 = nn.LayerNorm(h_dim)
        self.norm2 = nn.LayerNorm(h_dim)
        self.ffn = nn.Sequential(
            nn.Linear(h_dim, 4*h_dim),
            nn.GELU(),
            nn.Linear(4*h_dim, h_dim),
            nn.Dropout(drop_p)
        )

    def forward(self, x, key_padding_mask=None):
        # x: [B, T, H]
        # key_padding_mask: [B, T] with 1 for valid tokens, 0 for padding (optional)
        # Attention -> LayerNorm -> Residual -> FFN -> LayerNorm -> Residual
        out = self.norm1(x + self.attention(x, key_padding_mask))
        out = self.norm2(out + self.ffn(out))

        return out

# define the decision transformer
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_block, h_dim, context_len, n_heads, drop_p, max_timestep = 4096):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.context_len = context_len

        # transformer blocks
        input_seq_len = 3 * context_len
        blocks = [AttentionBlock(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_block)]
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