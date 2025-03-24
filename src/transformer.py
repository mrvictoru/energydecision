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

    def forward(self, x):
        # x: [B, T, H]
        B, T, C = x.shape # batch size, sequence length, hidden dimension * number of heads
        N, D = self.n_heads, C // self.n_heads # number of heads, dimension of each head

        # compute the query, key and value
        Q = self.Q_net(x).view(B, T, N, D).transpose(1, 2) # [B, N, T, D]
        K = self.K_net(x).view(B, T, N, D).transpose(1, 2)
        V = self.V_net(x).view(B, T, N, D).transpose(1, 2)

        # compute the attention
        weights = Q @ K.transpose(2,3) / math.sqrt(D) # QK^T / sqrt(D)
        weights = weights.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf')) # mask the future tokens 
        normalized_weights = F.softmax(weights, dim=-1) # softmax along the last dimension
        A = self.att_drop(normalized_weights @ V) # attention with dropout

        # compute the output
        # gather heads and project to correct dimension
        attention = A.transpose(1, 2).contiguous().view(B, T, N*D)
        out = self.proj_drop(self.proj_net(attention))

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

    def forward(self, x):
        # x: [B, T, H]
        # Attention -> LayerNorm -> Residual -> FFN -> LayerNorm -> Residual
        out = self.norm1(x + self.attention(x))
        out = self.norm2(out + self.ffn(out))

        return out

# define the decision transformer
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_block, h_dim, context_len, n_heads, drop_p, max_timestep = 4096):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        # transformer blocks
        input_seq_len = 3 * context_len
        blocks = [AttentionBlock(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_block)]
        self.transformer = nn.Sequential(*blocks)

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
    
    def forward(self, state, rtg, timestep, actions):
        B, T, _ = state.shape

        # timestep embedding
        time_emb = self.embed_timestep(timestep)

        # embedding for the state, reward and actions along with time embedding
        state_emb = self.embed_state(state) + time_emb
        rtg_emb = self.embed_rtg(rtg) + time_emb
        act_emb = self.embed_act(actions) + time_emb
        if act_emb.shape != time_emb.shape:
            act_emb = torch.squeeze(act_emb) # fix the unmatch dimension
        act_emb = act_emb + time_emb

        # stack the embeddings and reshape sequence as (r1, s1, a1, r2, s2, a2, ...)
        h = torch.stack([rtg_emb, state_emb, act_emb], dim=1).permute(0,2,1,3).reshape(B, 3*T, self.h_dim)
        h = self.embed_ln(h)

        # transformer blocks
        h = self.transformer(h)

        # get h reshaped such that its size is (B, 3, T, h_dim) and
        # h[:, 0, t] is conditioned on r_0, s_0, a_0, ..., r_t
        # h[:, 1, t] is conditioned on r_0, s_0, a_0, ..., r_t, s_t
        # h[:, 2, t] is conditioned on r_0, s_0, a_0, ..., r_t, s_t, a_t
        h = h.reshape(B, T, 3, self.h_dim).permute(0,2,1,3)

        # get predictions
        return_preds = self.pred_rtg(h[:,2])    # predict next rtg given r, s, a
        state_preds = self.pred_state(h[:,2])   # predict next state given r, s, a
        act_preds = self.pred_act(h[:,1])       # predict action given r, s

        return return_preds, state_preds, act_preds