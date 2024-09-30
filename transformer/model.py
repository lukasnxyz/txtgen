import torch
import torch.nn as nn
import torch.functional as F

# a communication mechanism
# notes:
# the keys and the values come from the encoder block if both encoding and decoding is implemented
# don't mask with tril in the encoder block, only in the decoder block, the mask prevents all 
#   nodes to talk with each other including future ones
class Attention(nn.Module):
  def __init__(self, head_size:int, block_size:int, n_embd:int):
    """
    Parameters
    ----------
    head_size 

    block_size 
      the number of tokens in a sequence
    n_embd
      the size of the vector representing a token in the embedding table
    """
    super().__init__()
    # what do I contain
    self.key = nn.Linear(n_embd, head_size, bias=False)
    # what am I looking for
    self.query = nn.Linear(n_embd, head_size, bias=False) 
    self.value = nn.Linear(n_embd, head_size, bias=False)
    # TODO: dropout
  
  def forward(self, x):
    # batch, time, channels
    # time is like the number of tokens (8 for block_size of 8)
    B, T, C = x.shape 
    k = self.key(x)
    q = self.query(x)
    # compute attention scores
    wei = q @ k.transpose(-2, -1) #* C**-0.5 # (B, T, 16) @ (B, 16, T) -> (B, T, T)
    tril = torch.tril(torch.ones(T, T)) # mask
    wei = wei.masked_file(tril == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)
    # weighted aggregation of the values
    v = self.value(x)
    out = wei @ v
    return out

class MultiHeadedAttention(nn.Module):
  def __init__(self):
    super().__init__()

# TODO: add input dims, middle dims are dims*4
# because paper 2048/512=4
class FeedForward(nn.Module):
  def __init__(self, dropout=0.1):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(),
      nn.ReLU(),
      nn.Linear(),
      # dropout
    )
  
  def forward(self, x):
    return self.net(x)
  
class Block(nn.Module):
  def __init__(self):
    super().__init__()

class Transformer(nn.Module):
  def __init__(self):
    super().__init__()