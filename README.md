# word2vec-numpy

A **pure NumPy implementation of Word2Vec** (Skip-gram with Negative Sampling) based on the 2013 Mikolov paper.

This project implements the full training pipeline from scratch:

* text preprocessing
* vocabulary creation
* skip-gram pair generation
* negative sampling
* forward pass, loss computation, and gradients
* parameter updates

No machine learning frameworks are required.

The model is trained on the **text8 dataset**, a cleaned Wikipedia corpus (~100M characters).

---

## Overview

Word2Vec learns vector representations of words such that words appearing in similar contexts end up **close in vector space**.

Training follows the **skip-gram with negative sampling (SGNS)** approach:

1. Pick a **center word** in the text.
2. Use surrounding words as **positive examples**.
3. Sample a few **negative words** randomly.
4. Update embeddings to bring center words closer to positive words and farther from negatives.

Over time, meaningful semantic and syntactic relationships emerge.

---

## Quick Setup

Clone the repository:

```bash
git clone https://github.com/<your-username>/word2vec-numpy.git
cd word2vec-numpy
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the **text8 dataset**:

```bash
mkdir -p data
wget http://mattmahoney.net/dc/text8.zip -P data/
unzip data/text8.zip -d data/
```

---

## Training

Run the training script:

```bash
python train.py
```

This will:

* preprocess the text
* build skip-gram pairs
* train the embeddings for 5 epochs
* save embeddings to `word2vec_embeddings.npy`

After training, **save the vocabulary**:

```bash
python save_vocab.py
```

This will create `vocab.pkl` required for the demo and evaluation scripts.

Default hyperparameters can be adjusted in `train.py`:

| Parameter                 | Default |
| ------------------------- | ------- |
| embedding dimension       | 100     |
| context window size       | 5       |
| negative samples per pair | 5       |
| epochs                    | 5       |
| learning rate             | 0.025   |
| minimum learning rate     | 1e-4    |
| minimum word frequency    | 5       |

---

## Demo and Results

All demos (nearest neighbors, word analogies, PCA visualizations, and cosine similarity heatmaps) are available in **`demo.ipynb`**.

Run the notebook to:

* load pre-trained embeddings
* inspect nearest neighbors
* evaluate analogies
* visualize embeddings with PCA
* compare trained embeddings against random vectors

```bash
jupyter notebook demo.ipynb
```

---

## Evaluating Embeddings

You can also use the evaluation functions directly in Python:

```python
import numpy as np
from evaluate import nearest_neighbors, analogy
import pickle

# Load embeddings
W = np.load("word2vec_embeddings.npy")

# Load saved vocabulary
with open("vocab.pkl", "rb") as f:
    data = pickle.load(f)
    vocab = data['vocab']
    idx2word = data['idx2word']

# Nearest neighbors
nearest_neighbors("king", vocab, idx2word, W)

# Word analogies
analogy("man", "king", "woman", vocab, idx2word, W)
```

---

## Project Structure

```text
word2vec-numpy/
├── word2vec.py      # model and training logic
├── train.py         # training script
├── save_vocab.py    # saves vocab.pkl
├── evaluate.py      # nearest neighbors & analogy evaluation
├── demo.ipynb       # notebook demonstrating embeddings
├── requirements.txt
└── data/            # dataset directory
```

---

## Implementation Notes

* **Two embedding matrices**: `W_in` for center words, `W_out` for context words. Only `W_in` is used as final embeddings.
* **Dynamic context window**: closer words get stronger training signal.
* **Subsampling frequent words**: reduces noise and speeds up training.
* **Efficient negative sampling**: uses a precomputed noise table.
* **Stable gradient updates**: handles duplicate negative samples with `np.add.at`.

---

## Requirements

```text
numpy>=1.24
matplotlib  # for demo
seaborn  #for demo
```

---

## References

* Tomas Mikolov et al., 2013 — [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
* Tomas Mikolov et al., 2013 — [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)

