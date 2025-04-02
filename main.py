import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms





class MultiLatentAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim, rope_factor):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.head_dim = embed_dim // num_heads
        self.embed_latent = nn.Linear(embed_dim, latent_dim)
        self.latent_embed = nn.Linear(latent_dim, embed_dim)
        self.q1_vector = nn.Parameter(torch.zeros(head_dim))
        self.k1_vector = nn.Parameter(torch.zeros(head_dim))
        self.q2_vector = nn.Parameter(torch.zeros(head_dim))
        self.k2_vector = nn.Parameter(torch.zeros(head_dim))
        self.factor = rope_factor
    def rope(x, pos):
        dim_size = x.size(-1)
        theta = 1 / (10000 ** (torch.arange(dim_size // 2).float() / (dim_size / 2))
        pos_theta = torch.einsum('i,j -> ij', pos, theta)
        sin, cosine = pos_theta.sin(), pos_theta.cos()
        first_half, second_half = x[..., :dim_size // 2], x[..., dim_size // 2:]
        return torch.cat([first_half * sin, second_half * cosine])
    def differentialAttentionHead(self, x, W_Q, W_K, W_V):
        query = torch.matmul(x, W_Q)
        key = torch.matmul(x, W_K)
        value = torch.matmul(x, W_V)
        q1, q2 = query[..., :query.size(-1) // 2], query[..., query.size(-1) // 2:]
        k1, k2 = key[..., :key.size(-1) // 2], key[..., key.size(-1) // 2:]
        s = 1 / sqrt(d)
        first_attention_mat = torch.matmul(q1, torch.matmul(torch.transpose(k1, −1, −2), s))
        second_attention_mat = torch.matmulI(q2, torch.matmul(torch.transpose(k2, −1, −2), s))
        return torch.matmul((first_attention_mat - self.lambda_vector * second_attention_mat), value)
   def forward(x):
       self.lambda_vector = torch.exp(torch.matmul(self.q1_vector, self.k1_vector)) - torch.exp(torch.matmul(self.q2_vector, self.k2_vector))
def main(epochs: int):
    discriminator = Discriminator()
    generator = Generator()
    dis_optimizer = optim.Adam(list(discriminator.parameters()))
    gen_optimizer = optim.Adam(list(generator.parameters()))
    for epoch in range(epochs):
        for i, (data, target) in enumerate(loader):
            gen_optimizer.zero_grad()
            image = generator(target)
            generator_criterion = nn.CrossEntropyLoss(discriminator(image), 1)
            generator_criterion.backward()
            gen_optimizer.step()
            dis_optimizer.zero_grad()
            discriminator_criterion = nn.CrossEntropyLoss(discriminator(data), 1)
            discriminator_criterion.backward()
            dis_optimizer.step()
            dis_optimizer.zero_grad()
            discriminator_criterion = nn.CrossEntropyLoss(discriminator(image), 0)
            discriminator_criterion.backward()
            dis_optimizer.step()
