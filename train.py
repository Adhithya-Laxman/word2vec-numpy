from word2vec import *

with open('data/text8', 'r') as f:
    raw = f.read()

tokens = tokenize(raw)
vocab, idx2word, counts = build_vocab(tokens, min_freq=5)

tokens = subsample(tokens, vocab, counts)
token_ids = [vocab[word] for word in tokens if word in vocab]

print("Vocabulary size:", len(vocab), "| Total tokens after subsampling:", len(token_ids))

pairs = build_pairs(token_ids, window_size=5)
print("Total training pairs:", len(pairs))
noise_table = build_noise_table(vocab, counts)

model = Word2Vec(vocab_size=len(vocab), embedding_dim=100)
model = train(model, pairs, noise_table, num_neg = 5, lr = 0.025, epochs=5)

np.save('word2vec_embeddings.npy', model.W_in)

