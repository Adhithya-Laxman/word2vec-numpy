import numpy as np
import re
from collections import Counter

####################################
# Preprocessing
####################################

def tokenize(text : str) -> list[str]:
    """Tokenizes the input text into a list of words."""
    # Convert to lowercase and split by non-alphanumeric characters
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def build_vocab(tokens, min_freq=1):
    counts = Counter(tokens)
    filtered = [(w, c) for w, c in counts.items() if c >= min_freq] 
    vocab    = {w: i for i, (w, c) in enumerate(filtered)}            
    idx2word = {i: w for w, i in vocab.items()}
    return vocab, idx2word, counts


def subsample(tokens : list[str], vocab : dict, counts : Counter, t: float = 1e-5) -> list[str]:
    """
        Mikolo et al. (2013) propose a subsampling technique to reduce the number of 
        training examples for frequent words.
        The probability of keeping a word is given by:
        P(w) = 1 - sqrt(t / freq(w))    
    """
    total = sum(counts.values())
    keep = {w: min(1.0, (np.sqrt(counts[w] / (t * total)) + 1) * (t * total / counts[w]))
            for w in vocab}    
    return [word for word in tokens if np.random.rand() < keep.get(word, 1)]

#####################################
# Skip-gram pair generation
#####################################

def build_pairs(token_ids: list[int], window_size: int) -> list[tuple[int, int]]:
    """Builds skip-gram pairs from a list of token IDs."""
    pairs = []
    for i, center in enumerate(token_ids):
        w = np.random.randint(1, window_size + 1)  # Random window size
        for j in range(max(0, i - w), min(len(token_ids), i + w + 1)):
            if j != i:
                pairs.append((center, token_ids[j]))
    return pairs

######################################
# Negative sampling table
######################################

def build_noise_table(vocab : dict, counts : Counter, table_size: int = 10_000_000, power: float = 0.75) -> np.ndarray:
    """
    Unigram distribution raised to the 3/4 power, as in the original paper.
    Stored as a flat array for O(1) sampling.
    """
    table = np.zeros(table_size, dtype=np.int32)
    total = sum(counts[word] ** power for word in vocab)
    idx, ptr = 0, 0
    for word, wid in vocab.items():
        ptr += (counts[word] ** power) / total
        while idx < table_size and idx / table_size < ptr:
            table[idx] = wid
            idx += 1
    return table


def sample_negative(table: np.ndarray, k: int, exclude: int = None) -> list[int]:
    """Samples k negative examples from the noise table, excluding a specific word ID."""
    negs = []
    while len(negs) < k:
        s = int(table[np.random.randint(len(table))])
        if s != exclude:
            negs.append(s)
    return negs

################################
# MODEL
################################

class Word2Vec:
    def __init__(self, vocab_size: int, embedding_dim: int, seed = 42):
        rng = np.random.default_rng(seed)
        # W_in - word embeddings, W_out - context embeddings
        self.W_in = rng.uniform(-0.5 / embedding_dim, 0.5 / embedding_dim, (vocab_size, embedding_dim))
        self.W_out = np.zeros((vocab_size, embedding_dim))  # Initialized to zero for negative sampling

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def forward_backward(self, center: int, context: int, negatives: list[int]) -> tuple[float, ...]:
        """
        Returns (loss, grad_v_c, grad_u_o, grad_U_neg).

        Derivation:
          J = -log σ(u_o·v_c) - Σ_k log σ(-u_k·v_c)
          ∂J/∂v_c  = (σ(u_o·v_c)-1)·u_o  + Σ_k σ(u_k·v_c)·u_k
          ∂J/∂u_o  = (σ(u_o·v_c)-1)·v_c
          ∂J/∂u_k  =  σ(u_k·v_c)·v_c
        """
        v_c = self.W_in[center]
        u_o = self.W_out[context]
        U_neg = self.W_out[negatives]

        pos_sig = self._sigmoid(u_o @ v_c)
        neg_sigs = self._sigmoid(-U_neg @ v_c)

        loss = -np.log(pos_sig + 1e-10) - np.sum(np.log(neg_sigs + 1e-10))

        grad_v_c = (pos_sig - 1) * u_o + np.sum((1-neg_sigs)[:, None] * U_neg, axis=0)
        grad_u_o = (pos_sig - 1) * v_c
        grad_U_neg = (1 - neg_sigs)[:, None] * v_c[None, :]
        return loss, grad_v_c, grad_u_o, grad_U_neg
    
    def step(self, center : int, context: int, negatives : list[int], grads: tuple[float, ...], lr: float):
        """Performs a single update step."""
        grad_v_c, grad_u_o, grad_U_neg = grads
        self.W_in[center] -= lr * grad_v_c
        self.W_out[context] -= lr * grad_u_o

        np.add.at(self.W_out, negatives, -lr * grad_U_neg)  # Handles duplicate negatives correctly

################################
# Training loop
################################

def train(model: Word2Vec, pairs: list[tuple[int, int]], noise_table: np.ndarray, num_neg: int = 5, lr: float = 0.025, min_lr: float = 1e-4, epochs : int = 5):
    """Trains the Word2Vec model using skip-gram with negative sampling."""
    total = len(pairs) * epochs
    processed = 0

    for epoch in range(1, epochs + 1):
        np.random.shuffle(pairs)
        epoch_loss = 0.0

        for center, context in pairs:
            # Linear decay of learning rate
            lr_t = max(min_lr, lr * (1 - processed / total))
            negatives = sample_negative(noise_table, num_neg, exclude=context)
            loss, *grads = model.forward_backward(center, context, negatives)
            model.step(center, context, negatives, grads, lr_t)
            epoch_loss += loss
            processed += 1
        print(f'Epoch {epoch}/{epochs}, Loss: {epoch_loss / len(pairs):.4f}, lr: {lr_t:.6f}')

    return model
    