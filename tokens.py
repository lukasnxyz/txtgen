# implementing a byte-pair tokenizer
from transformers import AutoTokenizer
from collections import defaultdict

def compute_pair_freqs(splits):
  pair_freqs = defaultdict(int)
  for word, freq in word_freqs.items():
    split = splits[word]
    if len(split) == 1: continue
    for i in range(len(split)-1):
      pair = (split[i], split[i+1])
      pair_freqs[pair] += freq
  return pair_freqs

tokenizer = AutoTokenizer.from_pretrained('gpt2')

with open('data/sample_truths.txt', 'r', encoding='utf-8') as f: 
  corpus = f.read().split('\n')

word_freqs = defaultdict(int)
for txt in corpus:
  wrds_w_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(txt)
  nw_wrds = [word for word, offset in wrds_w_offsets]
  for word in nw_wrds:
    word_freqs[word] += 1

#[print(f'{word}: {freq}') for word, freq in word_freqs.items()]

alphabet = []
for word in word_freqs.keys():
  for letter in word:
    if letter not in alphabet:
      alphabet.append(letter)
alphabet.sort()
chars = ['<|endoftext|>'] + alphabet.copy()
splits = {word: [c for c in word] for word in word_freqs.keys()}

pair_freqs = compute_pair_freqs(splits)
for i, key in enumerate(pair_freqs.keys()):
  print(f'{key}: {pair_freqs[key]}')
  if i >= 5: break