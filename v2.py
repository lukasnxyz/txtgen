import os
import requests
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

# download the tiny shakespeare dataset
input_file_path = os.path.join("data", "input.txt")
if not os.path.exists(input_file_path):
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open(input_file_path, "w", encoding="utf-8") as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, "r", encoding="utf-8") as f:
    data_txt = f.read()
# set torch device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")
# get chars from data
chars = sorted(list(set(data_txt)))
vocab_size = len(chars)
# encode/decode funcs
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda e: "".join([itos[i] for i in e])
# encode data
data = torch.tensor(encode(data_txt), dtype=torch.long)
# split data into train/val
n = int(0.9*len(data)) # 90%, 10%
train_data = data[:n]
val_data = data[n:]
# context window length and model batch size
ctx_len = 8
batch_size = 32
epochs = 5000
eval_interval = 1000
eval_iters = 200
n_embd = 32
learning_rate = 1e-3

# rnd seed
torch.manual_seed(42)

# get a random batch of chars from the data
def get_batch(split):
  data = train_data if split == "train" else val_data
  ix = torch.randint(len(data) - ctx_len, (batch_size,))
  x = torch.stack([data[i:i+ctx_len] for i in ix])
  y = torch.stack([data[i+1:i+ctx_len+1] for i in ix])
  return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ["train", "val"]:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(ctx_len, ctx_len)))
  
  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)
    wei = q @ k.transpose(-2, -1) * C**-0.5
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    # perform weighted aggregation of the values
    v = self.value(x)
    out = wei @ x
    return x
  
class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
  
  def forward(self, x):
    return torch.cat([h(x) for h in self.heads], dim=-1)

# bigram model  
class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(ctx_len, n_embd)
    self.sa_heads = MultiHeadAttention(4, n_embd//4) # i.e. 4 heads of 8-dimensional self-attention
    self.lm_head = nn.Linear(n_embd, vocab_size)
    
  def forward(self, idx, targets=None):
    B,T = idx.shape
    tok_emb = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device))
    x = tok_emb + pos_emb
    x = self.sa_heads(x)
    logits = self.lm_head(x) # (B,T,vocab_size)
    
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    
    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -ctx_len:]

      logits, loss = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

# -------------------

xb, yb = get_batch("train")
model = BigramLanguageModel().to(device)
logits, loss = model(xb, yb)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("------")
print("Before optimization:")
print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

for i in trange(epochs):
  if i % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  xb, yb = get_batch("train")
  
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

print("------")
print("After optimization:")
print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
