import numpy as np

def cosine_sim(a, b):
    dot    = a @ b.T                      
    norm_a = np.linalg.norm(a)            
    norm_b = np.linalg.norm(b, axis=1)    
    return dot / (norm_a * norm_b)        


def nearest_neighbors(word, vocab, idx2word, W, top_k=5):
    wid = vocab.get(word, None)
    if wid is None:
        print(f"Word '{word}' not found in vocabulary.")
        return []
    sims = cosine_sim(W[wid:wid+1], W).flatten()
    sims[wid] = -1 # Exclude the word itself

    for i in np.argsort(sims)[::-1][:top_k]:
        print(f"  {idx2word[i]:<15} {sims[i]:.4f}")
    
def analogy(a, b, c, vocab, idx2word, W, top_k=5):
    vec = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    vec /= np.linalg.norm(vec)
    sims = cosine_sim(vec[None], W).flatten()
    for excl in [vocab[a], vocab[b], vocab[c]]:
        sims[excl] = -1
    return idx2word[np.argmax(sims)]