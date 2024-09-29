### Text Generation
Implementations and experiments.
- https://arxiv.org/pdf/1706.03762
- https://arxiv.org/pdf/2409.10594

new repo name: attention
goal: just to play around with attention and different types of models

old:
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