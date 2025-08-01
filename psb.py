# SPDX-License-Identifier: Apache-2.0

from utils import *


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, num_heads, gain=1.0, dropout=0., inverted=False, epsilon=1e-5):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.inverted = inverted

        self.epsilon = epsilon
        
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        self.proj_q = linear(d_model, d_model, bias=False)
        self.proj_k = linear(d_model, d_model, bias=False)
        self.proj_v = linear(d_model, d_model, bias=False)
        self.proj_o = linear(d_model, d_model, bias=False, gain=gain)
    
    def forward(self, q, k, v, attn_mask=None, attn_bias=None):
        """
        q: batch_size x target_len x d_model
        k: batch_size x source_len x d_model
        v: batch_size x source_len x d_model
        attn_mask: target_len x source_len
        return: batch_size x target_len x d_model
        """
        B, T, _ = q.shape
        _, S, _ = k.shape
        
        q = self.proj_q(q).view(B, T, self.num_heads, -1).transpose(1, 2)
        k = self.proj_k(k).view(B, S, self.num_heads, -1).transpose(1, 2)
        v = self.proj_v(v).view(B, S, self.num_heads, -1).transpose(1, 2)
        
        q = q * (q.shape[-1] ** (-0.5))
        attn = torch.matmul(q, k.transpose(-1, -2))

        if attn_bias is not None:
            attn += attn_bias

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))

        if self.inverted:
            attn = F.softmax(attn.flatten(start_dim=1, end_dim=2), dim=1).reshape(B, self.num_heads, T, S)

            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask, float('0.'))

            attn /= attn.sum(dim=-1, keepdim=True) + self.epsilon
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_dropout(attn)
        
        output = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        output = self.proj_o(output)
        output = self.output_dropout(output)

        return output


class TemporalCompetitiveMultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, gain=1.0, dropout=0., epsilon=1e-5):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads

        self.epsilon = epsilon

        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        self.proj_q = linear(d_model, d_model, bias=False)
        self.proj_k = linear(d_model, d_model, bias=False)
        self.proj_v = linear(d_model, d_model, bias=False)
        self.proj_o = linear(d_model, d_model, bias=False, gain=gain)

    def forward(self, q, k, v, attn_mask=None, attn_bias=None):
        """
        q: B, T, N, D
        k: B, T, L, D
        v: B, T, L, D
        attn_mask: TN x TL
        attn_bias: B, H, TN, TL
        return: B, T, N, D
        """
        B, T, N, _ = q.shape
        _, _, L, _ = k.shape

        q = self.proj_q(q).view(B, T * N, self.num_heads, -1).transpose(1, 2)
        k = self.proj_k(k).view(B, T * L, self.num_heads, -1).transpose(1, 2)
        v = self.proj_v(v).view(B, T * L, self.num_heads, -1).transpose(1, 2)

        q = q * (q.shape[-1] ** (-0.5))
        attn = torch.matmul(q, k.transpose(-1, -2))

        if attn_bias is not None:
            attn += attn_bias

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))

        attn = attn.reshape(B, self.num_heads, T, N, T * L)
        attn = attn.permute(0, 2, 4, 1, 3).reshape(B, T, T * L, self.num_heads * N)
        attn = F.softmax(attn, dim=-1).reshape(B, T, T * L, self.num_heads, N).permute(0, 3, 1, 4, 2)
        attn = attn.reshape(B, self.num_heads, T * N, T * L)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('0.'))

        attn /= attn.sum(dim=-1, keepdim=True) + self.epsilon

        attn = self.attn_dropout(attn)

        output = torch.matmul(attn, v).transpose(1, 2).reshape(B, T * N, -1)
        output = self.proj_o(output)
        output = self.output_dropout(output)

        output = output.reshape(B, T, N, -1)

        return output


class PSBBlock(nn.Module):

    def __init__(self, d_model, num_heads, max_len=1024, gain=1.0, dropout=0.0, relpos_scalar_dim=16, relpos_mlp_dim=128, num_heads_bu=1):
        super().__init__()

        self.max_len = max_len
        self.num_heads = num_heads
        self.num_heads_bu = num_heads_bu

        self.relpos_mlp = nn.Sequential(
            OctavesScalarEncoder(relpos_scalar_dim, 2. * max_len - 1.),
            linear(relpos_scalar_dim, relpos_mlp_dim, weight_init='kaiming'),
            nn.GELU(),
            linear(relpos_mlp_dim, num_heads_bu + num_heads),
        )

        self.causal_mask = nn.Parameter(torch.triu(torch.ones((max_len, max_len), dtype=torch.bool), diagonal=1), requires_grad=False)

        self.cross_attn = TemporalCompetitiveMultiHeadAttention(d_model, num_heads_bu, gain=gain, dropout=dropout)

        self.self_attn_across_time = MultiHeadAttention(d_model, num_heads, gain=gain, dropout=dropout)

        self.self_attn = MultiHeadAttention(d_model, num_heads, gain=gain, dropout=dropout)

        self.ffn = nn.Sequential(
            linear(d_model, 4 * d_model, weight_init='kaiming'),
            nn.GELU(),
            linear(4 * d_model, d_model, gain=gain),
            nn.Dropout(dropout)
        )

        self.norm_cross_attn_input = nn.LayerNorm(d_model)
        self.norm_cross_attn_cond = nn.LayerNorm(d_model)
        self.norm_self_attn_across_time = nn.LayerNorm(d_model)
        self.norm_self_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, input, cond):
        """
        input: B, T, N, D
        cond: B, T, L, D
        return: B, T, N, D
        """

        B, T, N, D = input.shape
        _, _, L, _ = cond.shape

        indices = torch.arange(T, device=input.device)  # T
        indices = indices[None, :] - indices[:, None]  # T, T

        # relative positional embeddings
        relpos_bias = self.relpos_mlp(indices)  # T, T, 2 * num_heads
        relpos_bias_1, relpos_bias_2 = relpos_bias.split([self.num_heads_bu, self.num_heads], dim=-1)  # T, T, num_heads_bu; T, T, num_heads
        relpos_bias_1 = relpos_bias_1[None, :, None, :, None, :].repeat(1, 1, N, 1, L, 1)  # 1, T, N, T, L, num_heads_bu
        relpos_bias_1 = relpos_bias_1.reshape(1, T * N, T * L, -1).permute(0, 3, 1, 2)  # 1, num_heads_bu, TN, TL
        relpos_bias_2 = relpos_bias_2[None, :, :, :].permute(0, 3, 1, 2)  # 1, num_heads, T, T

        # causal mask
        causal_mask_1 = self.causal_mask[:T, None, :T, None].repeat(1, N, 1, L).reshape(T * N, T * L)  # TN, TL
        causal_mask_2 = self.causal_mask[:T, :T]  # T, T

        # slots attend view features
        cond = self.norm_cross_attn_cond(cond)  # B, T, L, D
        x = self.norm_cross_attn_input(input)  # B, T, N, D
        input = input + self.cross_attn(x, cond, cond, attn_bias=relpos_bias_1, attn_mask=causal_mask_1)  # B, T, N, D

        # slots attend other time-step slots having same slot index
        input = input.permute(0, 2, 1, 3).reshape(B * N, T, D)  # BN, T, D
        x = self.norm_self_attn_across_time(input)  # BN, T, D
        input = input + self.self_attn_across_time(x, x, x, attn_bias=relpos_bias_2, attn_mask=causal_mask_2)  # BN, T, D
        input = input.reshape(B, N, T, D).permute(0, 2, 1, 3)  # B, T, N, D

        # slots attend same time-step slots
        input = input.reshape(B * T, N, D)  # BT, N, D
        x = self.norm_self_attn(input)  # BT, N, D
        input = input + self.self_attn(x, x, x)  # BT, N, D
        input = input.reshape(B, T, N, D)  # B, T, N, D

        # ffn
        x = self.norm_ffn(input)  # B, T, N, D
        input = input + self.ffn(x)  # B, T, N, D

        return input  # B, T, N, D


class PSB(nn.Module):

    def __init__(self, d_model, num_blocks, num_heads, dropout=0.0):
        super().__init__()

        self.d_model = d_model

        self.norm = nn.LayerNorm(d_model)

        gain = (4 * num_blocks) ** (-0.5)
        self.blocks = nn.ModuleList(
            [PSBBlock(d_model, num_heads, gain=gain, dropout=dropout)
             for _ in range(num_blocks)])

    def forward(self, slots, input):
        """
        input: B, T, N, D
        cond: B, T, L, D
        return: B, T, N, D
        """

        slots = self.norm(slots)
        for block in self.blocks:
            slots = block(slots, input)

        return slots
