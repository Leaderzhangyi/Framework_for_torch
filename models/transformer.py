from torch import nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3*embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size = x.size(0)
        qkv = self.qkv_proj(x).reshape(batch_size, -1, self.num_heads, 3*self.head_dim).transpose(1, 2)
        # print("qkv shape:", qkv.shape)  # 打印 qkv 的形状
        q, k, v = qkv.chunk(3, dim=-1)
        # print("q shape:", q.shape)  # 打印 q 的形状
        q = q / self.head_dim ** 0.5
        scores = (q @ k.transpose(-2, -1))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x) + residual
        x = self.dropout(x)
        residual = x
        x = self.norm2(x)
        x = self.feedforward(x) + residual
        x = self.dropout(x)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, feedforward_dim, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_layers, num_heads, feedforward_dim, dropout):
        super(TransformerModel, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.encoder = TransformerEncoder(embed_dim, num_layers, num_heads, feedforward_dim, dropout)
        self.out_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.out_proj(x)
        return x