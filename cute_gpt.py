from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

#
@dataclass
class GPTconfig:
    block_size: int =256
    vocab_size: int =65
    n_layer: int = 6
    n_head: int =6
    n_embed: int = 384

class CasualSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed, 3*config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        #multi-head attention:多个头并行计算Q,K,V,生成的结果拼接即可
        self.n_head=config.n_head
        self.n_embed=config.n_embed
        #缓冲区是一种特殊的张量，它不是可训练的参数（即不会通过梯度更新），但会在模型中存储并随着模型的状态一起保存。
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size,config.block_size)).
                             view(1,1,config.block_size,config.block_size))
        
    def forward(self,x):
        B,T,C=x.size()  #batch size, sequence length , n_embed
        qkv=self.c_attn(x)
        q,k,v=qkv.split(self.n_embed,dim=2)
        k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        #attention
        att = (q @ k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
        att = F.softmax(att,dim=-1)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)   #just concatenate
        y=self.c_proj(y)

        return y



class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4* config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embed,config.n_embed)
    
    def foward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x



class Block(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embed)
        self.attn=CasualSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embed)
        self.mlp=MLP(config)
    def forward(self,x):
        x=x+self.attn(self.ln_1(x))
        x=x+self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.config=config

        self.transformer = nn.ModuleDict({
            "wte":nn.Embedding(config.vocab_size, config.n_embed),
            "wpe":nn.Embedding(config.block_size, config.n_embed),
            "h":nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f":nn.LayerNorm(config.n_embed)

        })
        self.lm_head=nn.Linear(config.n_embed, config.vocab_size, bias=False)
