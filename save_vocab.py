# save_vocab.py  ← create this file, run once
import pickle
from word2vec import *

with open('data/text8', 'r') as f:
    raw = f.read()

tokens = tokenize(raw)
vocab, idx2word, counts = build_vocab(tokens, min_freq=5)

with open('vocab.pkl', 'wb') as f:
    pickle.dump({'vocab': vocab, 'idx2word': idx2word, 'counts': counts}, f)

print(f"Saved vocab.pkl | Vocab size: {len(vocab):,}")
