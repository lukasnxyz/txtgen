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
print(f"chars: {''.join(chars)}")
print(f"num of chars: {vocab_size}")
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
epochs = 10000
eval_interval = 1000
eval_iters = 200
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

# bigram model  
class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
  def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx) # (B,T,vocab_size)
    
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
      logits, loss = self(idx)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

# -------------------

xb, yb = get_batch("train")
model = BigramLanguageModel().to(device)
logits, loss = model(xb, yb)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

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