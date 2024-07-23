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
* train loss 2.9713, val loss 3.5299
* train loss 1.8865, val loss 7.1341 (19.4m on mba m3 16gb)

### To run in Colab
```python
!wget 'https://raw.githubusercontent.com/lukasnxyz/txttokens/main/tokens.py'
!wget 'https://raw.githubusercontent.com/lukasnxyz/txtgen/main/model/transformer.py'
!wget 'https://raw.githubusercontent.com/lukasnxyz/txtgen/main/model/utils.py'
!wget 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
```
