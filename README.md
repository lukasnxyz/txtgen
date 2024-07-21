### Text Generation

### Quick start
```bash
python3 -m venv venv
pip3 install -r requirements.txt
```
download datasets
```bash
python3 get_data.py
```

### Stats
train loss 2.9713, val loss 3.5299

### To run in Colab
```python
!wget "https://raw.githubusercontent.com/lukasnxyz/txttokens/main/tokens.py"
!wget "https://raw.githubusercontent.com/lukasnxyz/txtgen/main/model/transformer.py"
!wget "https://raw.githubusercontent.com/lukasnxyz/txtgen/main/model/utils.py"

import requests
with open('truths.txt', 'w', encoding='utf-8') as f:
    f.write(requests.get('https://gist.githubusercontent.com/lukasnxyz/d7b29398dd3b3d1c1dcb14f1f19e3744/raw/8157e9b21d4169d39a99bd8e27287a5bc44de31a/truth_sentences.txt').text)
```
