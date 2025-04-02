import torch
import torch.nn as nn
import torch.optim as optim
kernel_size = 3
embedding_dim = 256
num_heads = 25
rope_factor = 12
latent_dim = 128
residual_stream = torch.zeros(embed_dim)
dim_size = 64
class MultiLatentAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim, rope_factor):
        torch.utils.checkpoint.CheckpointPolicy(1)
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.lambda_init = torch.zeros(head_dim)
        self.head_dim = embed_dim // num_heads
        self.embed_latent = nn.Linear(embed_dim, latent_dim)
        self.latent_embed = nn.Linear(latent_dim, embed_dim)
        self.key_cache = torch.zeros(latent_dim)
        self.value_cache = torch.zeros(latent_dim)
        self.q1_vector = nn.Parameter(torch.zeros(head_dim))
        self.k1_vector = nn.Parameter(torch.zeros(head_dim))
        self.q2_vector = nn.Parameter(torch.zeros(head_dim))
        self.k2_vector = nn.Parameter(torch.zeros(head_dim))
        self.projectionMatrice = nn.Linear(embed_dim, latent_dim // 16)
        self.factor = rope_factor
    def rope(x):
        dim_size = x.size(-1)
        theta = 1 / (10000 ** (torch.arange(dim_size // 2).float() / (dim_size / 2))
        pos_theta = torch.einsum('i,j -> ij', x.size(-1), theta)
        return torch.cat([x[..., :dim_size // 2] * pos_theta.sin(), x[..., dim_size // 2:] * pos_theta.cos()])
    def differentialAttentionHead(self, x, W_Q, W_K, W_V):
        self.key_cache = torch.cat([self.key_cache, self.rope(self.key_cache)])
        self.key_cache = self.latent_embed(self.key_cache)
        self.value_cache = self.latent_embed(self.value_cache)
        query = self.projectionMatrice(torch.sparse.mm(x, W_Q))
        key = self.projectionMatrice(torch.sparse.mm(x, W_K))
        value = self.projectionMatrice(torch.sparse.mm(x, W_V))
        key = torch.cat([key, self.rope(key)])
        self.k_cache = torch.cat([self.k_cache, key], dim=2) 
        self.v_cache = torch.cat([self.v_cache, value], dim=2)
        q1, q2 = query[..., :query.size(-1) // 2], query[..., query.size(-1) // 2:]
        k1, k2 = self.k_cache[..., :self.k_cache.size(-1) // 2], self.k_cache[..., self.k_cache.size(-1) // 2:]
        s = 1 / sqrt(d)
        first_attention_mat = torch.sparse.mm(q1, torch.sparse.mm(torch.transpose(k1, −1, −2), s))
        second_attention_mat = torch.sparse.mm(q2, torch.sparse.mm(torch.transpose(k2, −1, −2), s))
        output = torch.matmul((first_attention_mat - self.lambda_vector * second_attention_mat), self.v_cache)
        self.v_cache = self.embed_latent(self.v_cache)
        self.k_cache = self.embed_latent(self.k_cache)
        return output
   def MultiHeadAttention(self, x, W_Q, W_K, W_V, result_weight):
        result = nn.GroupNorm([differentialAttentionHead(x, W_Q, W_K, W_V) for i in range(self.num_heads)]
        result = torch.matmul(result, (1 − self.lambda_init))
        return torch.cat(result, result_weight)
   def forward(x):
       self.lambda_vector = torch.exp(torch.matmul(self.q1_vector, self.k1_vector)) - torch.exp(torch.matmul(self.q2_vector, self.k2_vector)) + self.lambda_init
class Generator(nn.Module):
    def __init__(self, embedding_size: int, dim_size: int, kernel_size: int):
        super(Generator, self).__init__()
        self.denoiser = nn.Sequential(
            nn.Conv2d(embedding_size, dim_size, kernel_size, padding=kernel_size//2),
            nn.ReLU(True),
            nn.Conv2d(dim_size, dim_size * 2, kernel_size * 2, padding=kernel_size),
            nn.ReLU(True)
        )
        self.transposer = nn.Sequential(
            nn.ConvTranspose2d(dim_size * 2, dim_size, kernel_size * 2, padding=kernel_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim_size, embedding_size, kernel_size, padding=kernel_size//2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(embedding_size * 28 * 28, dim_size),
            nn.Softmax(dim=1)
        )
    def forward(self, x, min_val, max_val):
        denoised = self.denoiser(x)
        transposed = self.transposer(denoised)
        return torch.clamp(transposed, min=min_val, max=max_val)
