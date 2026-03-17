import numpy as np

def cosine_sim(a : np.ndarray, b : np.ndarray) -> np.ndarray:
    """Computes cosine similarity between a vector and a matrix of vectors.
        a: shape (d,)
        b: shape (N, d)
    Returns:
        sims: shape (N,) cosine similarities    
    """

    dot    = a @ b.T                      
    norm_a = np.linalg.norm(a)            
    norm_b = np.linalg.norm(b, axis=1)    
    return dot / (norm_a * norm_b)        


def nearest_neighbors(word : str, vocab : dict, idx2word : dict, W : np.ndarray, top_k : int = 5) -> list[str]:
    """
    Prints the top_k nearest neighbors of a given word based on cosine similarity.
    Args:        
        word: The query word.
        vocab: A dictionary mapping words to their indices in the embedding matrix.
        idx2word: A dictionary mapping indices back to words.
        W: The embedding matrix of shape (vocab_size, embedding_dim).
        top_k: The number of nearest neighbors to return.
    Returns:        
        A list of the top_k nearest neighbor words. 
    """
    wid = vocab.get(word, None)
    if wid is None:
        print(f"Word '{word}' not found in vocabulary.")
        return []
    sims = cosine_sim(W[wid:wid+1], W).flatten()
    sims[wid] = -1 # Exclude the word itself

    for i in np.argsort(sims)[::-1][:top_k]:
        print(f"  {idx2word[i]:<15} {sims[i]:.4f}")
    
def analogy(a: str, b: str, c: str, vocab: dict, idx2word: dict, W: np.ndarray, top_k: int = 5) -> str:
    """
    Solves the analogy task: a is to b as c is to ?
    Args:
        a, b, c: The three words in the analogy (e.g., "king", "queen", "man").
        vocab: The word-to-index mapping.
        idx2word: The index-to-word mapping.
        W: The embedding matrix.
        top_k: The number of top candidates to return.
    Returns:    
        The word that best completes the analogy.
    """
    vec = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    vec /= np.linalg.norm(vec)
    sims = cosine_sim(vec[None], W).flatten()
    for excl in [vocab[a], vocab[b], vocab[c]]:
        sims[excl] = -1
    return idx2word[np.argmax(sims)]