import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class MultiLatentAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim, rope_factor):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.head_dim = embed_dim // num_heads
        self.factor = rope_factor
        self.WQ = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.WK = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.WV = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.result_weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.lambda_init = nn.Parameter(torch.zeros(1))
        self.embed_latent = nn.Linear(embed_dim, latent_dim)
        self.latent_embed = nn.Linear(latent_dim, embed_dim)
        self.key_cache = None
        self.value_cache = None
        self.q1_vector = nn.Parameter(torch.randn(embed_dim))
        self.k1_vector = nn.Parameter(torch.randn(embed_dim))
        self.q2_vector = nn.Parameter(torch.randn(embed_dim))
        self.k2_vector = nn.Parameter(torch.randn(embed_dim))
        self.projectionMatrice = nn.Linear(embed_dim, latent_dim // 16)
    def rope(self, x):
        dim_size = x.size(-1)
        theta = 1 / (10000 ** (torch.arange(0, dim_size // 2).float() / (dim_size // 2)))
        theta = theta.to(x.device)
        pos = torch.arange(0, x.size(-2), device=x.device).float()
        pos_theta = torch.einsum('i,j->ij', pos, theta)
        sin_part = x[..., :dim_size // 2] * pos_theta.sin()
        cos_part = x[..., dim_size // 2:] * pos_theta.cos()
        return torch.cat([sin_part, cos_part], dim=-1)
    def differential_attention_head(self, x):
        query = self.projectionMatrice(torch.matmul(x, self.WQ))
        key = self.projectionMatrice(torch.matmul(x, self.WK))
        value = self.projectionMatrice(torch.matmul(x, self.WV))
        if self.key_cache is None:
            self.key_cache = key
            self.value_cache = value
        else:
            self.key_cache = torch.cat([self.key_cache, key], dim=1)
            self.value_cache = torch.cat([self.value_cache, value], dim=1)
        q1, q2 = query.chunk(2, dim=-1)
        k1, k2 = self.key_cache.chunk(2, dim=-1)
        scale = 1 / math.sqrt(self.head_dim)
        attn_1 = torch.matmul(q1, k1.transpose(-1, -2)) * scale
        attn_2 = torch.matmul(q2, k2.transpose(-1, -2)) * scale
        lambda_vector = torch.exp(torch.dot(self.q1_vector, self.k1_vector)) - torch.exp(torch.dot(self.q2_vector, self.k2_vector)) + self.lambda_init
        output = torch.matmul((attn_1 - lambda_vector * attn_2), self.value_cache)
        return output
    def forward(self, x):
        heads = [self.differential_attention_head(x) for _ in range(self.num_heads)]
        concatenated = torch.cat(heads, dim=-1)
        result = torch.matmul(concatenated, self.result_weight)
        return result
class Generator(nn.Module):
    def __init__(self, embedding_size: int, dim_size: int, kernel_size: int):
        super().__init__()
        self.denoiser = nn.Sequential(
            nn.Conv2d(embedding_size, dim_size, kernel_size, padding=kernel_size // 2),
            nn.ReLU(True),
            nn.Conv2d(dim_size, dim_size * 2, kernel_size, padding=kernel_size // 2),
            nn.ReLU(True)
        )
        self.transposer = nn.Sequential(
            nn.ConvTranspose2d(dim_size * 2, dim_size, kernel_size, padding=kernel_size // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim_size, embedding_size, kernel_size, padding=kernel_size // 2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(embedding_size * 28 * 28, dim_size),
            nn.Softmax(dim=1)
        )
    def forward(self, x, min_val=0, max_val=1):
        denoised = self.denoiser(x)
        transposed = self.transposer(denoised)
        return torch.clamp(transposed, min=min_val, max=max_val)
class Model(nn.Module):
    def __init__(self, kernel_size, embedding_dim, num_heads, rope_factor, latent_dim, dim_size, vocab_size):
        super().__init__()
        self.head_dim = embedding_dim // num_heads
        self.attn = MultiLatentAttention(embedding_dim, num_heads, latent_dim, rope_factor)
        self.generator = Generator(embedding_dim, dim_size, kernel_size)
        self.classifier = nn.Linear(dim_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x, layers=1):
        out = x
        for _ in range(layers):
            out = self.attn(out)
            out = self.generator(out)
            out = self.softmax(self.classifier(out))
        return out
