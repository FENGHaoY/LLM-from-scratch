import torch
import torch.nn as nn
from torchinfo import summary

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4*cfg['emb_dim']),
            nn.GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
        )
    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,num_heads,dropout,qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # 必须是整除
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(in_features=d_in, out_features=d_out,bias=qkv_bias)
        self.W_k = nn.Linear(in_features=d_in, out_features=d_out,bias=qkv_bias)
        self.W_v = nn.Linear(in_features=d_in, out_features=d_out,bias=qkv_bias)
        self.out_proj = nn.Linear(in_features=d_out, out_features=d_out)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),diagonal=1)
        )
        
    def forward(self, x):
        b, num_tokens, d_in =x.shape
        queries = self.W_q(x).view(b, num_tokens, self.num_heads, self.head_dim)
        keys = self.W_k(x).view(b, num_tokens, self.num_heads, self.head_dim)
        values = self.W_v(x).view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.transpose(1,2) # [b, num_heads, num_tokens, head_dim]
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        atten_scroe = queries @ keys.transpose(2,3) / keys.shape[-1] ** 0.5
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        atten_scroe.masked_fill_(mask_bool, -torch.inf)
        atten_weight = torch.softmax(atten_scroe, dim=-1)
        atten_weight = self.dropout(atten_weight)
        content_vec = atten_weight @ values# [b, num_heads, num_tokens, head_dim]
        content_vec = content_vec.transpose(1,2).contiguous().view(b, num_tokens, self.d_out)# [b, num_tokens, d_out]
        content_vec = self.out_proj(content_vec)
        return content_vec

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(cfg['emb_dim'],
                                      cfg['emb_dim'],
                                      cfg['context_length'],
                                      cfg['n_heads'],
                                      cfg['drop_rate'],
                                      qkv_bias=cfg['qkv_bias'])
        self.ff = FeedForward(cfg)
        self.LN1 = nn.LayerNorm(cfg['emb_dim'])
        self.LN2 = nn.LayerNorm(cfg['emb_dim'])
        self.dropout = nn.Dropout(cfg['drop_rate'])
    
    def forward(self,x):
        shoutcut = x # 后面用于残差连接
        x = self.LN1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + shoutcut
        shoutcut = x
        x = self.LN2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shoutcut
        return x
    
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_norm = nn.Dropout(cfg['drop_rate'])
        self.trf_blocks = nn.Sequential(
            *[ TransformerBlock(cfg) for _ in range (cfg['n_layers']) ])
        self.final_norm = nn.LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'],bias=False)
    
    def forward(self, in_idx):
        batch, seq_len = in_idx.shape
        token = self.token_emb(in_idx)
        pos = self.pos_emb(torch.arange(seq_len,device=in_idx.device))#key torch.arange默认用cpu
        x = token + pos
        x = self.drop_norm(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits 

if __name__ == '__main__':
    GPT_CONFIG_124M = {
    'vocab_size': 50257,
    'context_length': 1024,
    'emb_dim': 768,
    'n_heads':12,
    'n_layers':12,
    'drop_rate':0.1,
    'qkv_bias':False
}
    model = GPTModel(cfg=GPT_CONFIG_124M)
    print(sum(p.numel() for p in model.parameters()))
    summary(model, input_size=(2, 4), dtypes=[torch.long])
    device = torch.device('cuda')
    model.to(device)
        # ✅ 构造模拟输入：整数 token 索引
    dummy_input = torch.randint(0, 50257, (2, 4)).to(device)  # (batch=2, seq_len=4)

    # ✅ 测试前向传播
    output = model(dummy_input)
    print("输出 shape:", output.shape)
    