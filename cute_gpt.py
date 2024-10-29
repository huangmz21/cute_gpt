from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
import tiktoken

os.environ["CUDA_VISIBLE_DEVICES"]="7"

#
@dataclass
class GPTconfig:
    block_size: int =1024
    vocab_size: int =50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768

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
        #所以这个attn.bias是一个固定的值，我们不需进行存储或copy
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
    
    def forward(self,x):
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
    
    def forward(self,idx,targets= None):
        #idx shape(B,T)
        B,T=idx.size()
        assert T<= self.config.block_size
        #这里的embedding是一个查找表，输入indices，
        #设indices为[1,2,3],ouput=[embed[1,:],...],所以扩大了矩阵维度,保证device一致
        pos = torch.arange(0,T,dtype=torch.long,device=idx.device)
        pos_emb = self.transformer.wpe(pos)  #(T,n_embed)
        tok_emb = self.transformer.wte(idx)  #(B,T,n_embed)
        x = tok_emb + pos_emb
        #
        for block in self.transformer.h:
            x=block(x)
        x=self.transformer.ln_f(x)
        logits = self.lm_head(x)  #(B,T,vocab_size)
        loss = None
        if targets is not  None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits, loss
            

        
    @classmethod
    def from_pretrained(cls,model_type):
        "loads from huggingface"
        assert model_type in {'gpt2','gpt2-medium','gpt2-large','gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from %s" %model_type)

        config_args = {
            'gpt2':     dict(n_layer=12, n_head=12, n_embed=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embed=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embed=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embed=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTconfig(** config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard the mask

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        #copy while all parameter are aligned
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight','attn.c_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatah keys: {len(sd_keys)!=len(sd_keys_hf)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                #as the conv1D is (indim,outdim),while Linear_matrix is (outdim,indim)
                #nn的1维卷积Conv1(in_channel,out_channel,kernerl),相当于有out_channel个(in_channel*kernel)大小的卷积核
                #而输入为(batch,in_channel,embed),由此可见kernel只能在一个方向上滑动，因此被称为1Dconv
                #所以输出为(batch,out_channel,...)
                #这个地方我下意识以为conv1D是由torch.nn调用的，结果完全对不上，查看site-packages/transformers/models/gpt2/modeling_gpt2.py后得知是自己定义的，引以为戒
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

#model = GPT.from_pretrained('gpt2')
#print("didn't fail")

#demo------------
num_return_sequences = 5
max_length = 30
device = 'cuda'
#model=GPT.from_pretrained('gpt2')
model = GPT(GPTconfig())
model.eval()
model.to(device)

#prefix tokens
enc=tiktoken.get_encoding('gpt2')
tokens = enc.encode("hi,I am an LLM")
tokens = torch.tensor(tokens, dtype = torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)
x = tokens.to(device)

# generate! now x is (B,T)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits ,loss = model(x) #(B,T,vocab_size)
        ##take the logits at the last
        logits = logits[:,-1,:]
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50
        # so the topk_prob is (5,50),可能生成的token在0-50256，所以porbs是概率值，indices对应token的位置
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token,使用概率值进行采样
        ix = torch.multinomial(topk_probs,1)  # (B,1)
        xcol = torch.gather(topk_indices,-1,ix) #(B,1)
        # append to the sequence
        x = torch.cat((x,xcol),dim=1)

#print the generated results
for i in range(num_return_sequences):
    tokens=x[i,:max_length].tolist()
    decoded = enc.decode(tokens)
    print(">>",decoded)

#-----trainging process
#dataloader
enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()
text = text[:1000]
tokens = enc.encode(text)
B, T =4, 32
buf = torch.tensor(tokens[:B*T+1]).to(device)
x = buf[:-1].view(B,T)
y = buf[1:].view(B,T)
#loss
model= GPT(GPTconfig())
model.to(device)
logits, loss= model(x, y)  #用了重载函数
#optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for i in range(50):
    optimizer.zero_grad()
    logits, loss = model(x,y)
    loss.backward()
    optimizer.step()
    print(f"step{i}, loss{loss.item()}")




