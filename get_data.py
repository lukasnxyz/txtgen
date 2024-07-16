import os
import requests

print('choose data to download:')
print('0. names\n1. tiny_shakespeare\n2. truth_sentences')
choice = int(input('> '))

data_dir = 'data/'
if not os.path.exists(data_dir): 
    print(f'making dir: {data_dir}')
    os.makedirs(data_dir)

urls = ['https://raw.githubusercontent.com/karpathy/makemore/master/names.txt',
        'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        'https://gist.githubusercontent.com/lukasnxyz/d7b29398dd3b3d1c1dcb14f1f19e3744/raw/8157e9b21d4169d39a99bd8e27287a5bc44de31a/truth_sentences.txt']

match choice:
    case 0: 
        in_path = os.path.join(data_dir, 'names.txt')
        if not os.path.exists(in_path):
            with open(in_path, 'w', encoding='utf-8') as f: 
                print(f'downloading {urls[choice]}')
                f.write(requests.get(urls[choice]).text)
    case 1: 
        in_path = os.path.join(data_dir, 'tiny_shakespeare.txt')
        if not os.path.exists(in_path):
            with open(in_path, 'w', encoding='utf-8') as f: 
                print(f'downloading {urls[choice]}')
                f.write(requests.get(urls[choice]).text)
    case 2: 
        in_path = os.path.join(data_dir, 'truth_sentences.txt')
        if not os.path.exists(in_path):
            with open(in_path, 'w', encoding='utf-8') as f: 
                print(f'downloading {urls[choice]}')
                f.write(requests.get(urls[choice]).text)
    case _:
        print(f'input {choice} is invalid!')