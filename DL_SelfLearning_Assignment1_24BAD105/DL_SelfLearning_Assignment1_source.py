"""Source-code companion for DL Self-Learning Assignment 1.

Open the .ipynb notebook in Google Colab for markdown instructions and plots.

In Colab, the %pip cell is supported as notebook magic; remove it or run pip

from a shell before using this file as a plain .py script.
"""

# Run separately in a notebook or terminal: pip install "gensim>=4.3.3" "transformers>=4.40.0,<5"


# %%
import os, re, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import scipy.linalg
# Compatibility patch for environments where SciPy no longer exposes triu.
if not hasattr(scipy.linalg, "triu"):
    scipy.linalg.triu = np.triu

from gensim.models import Word2Vec, FastText
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
print("Python:", sys.version.split()[0])
print("TensorFlow:", tf.__version__)
print("NumPy:", np.__version__)


# %%
corpus = [
    # Technology
    "python code helps students build software applications",
    "computer systems process data using algorithms and programs",
    "machine learning models learn patterns from training data",
    "the neural network classifies text and predicts a label",
    "software developers debug code and test a computer program",
    "data science combines statistics programming and visualization",
    "the algorithm processes information stored in a database",
    "a computer uses memory and a processor to run software",
    "artificial intelligence uses data to make useful predictions",
    "programming students practice python and solve coding problems",
    "a digital system can store retrieve and analyze information",
    "the model improves when it is trained with more examples",
    # Sports
    "basketball players practice shooting passing and defending",
    "the coach trains the team before the basketball game",
    "athletes exercise daily to improve strength speed and fitness",
    "the player scored points on the basketball court",
    "football teams use tactics to win important matches",
    "a coach studies game performance and player statistics",
    "training helps an athlete build muscle and endurance",
    "the team celebrated after winning the final game",
    "sports improve physical health teamwork and discipline",
    "players practice basketball drills at the sports ground",
    "the match referee checks the score and follows the rules",
    "a player needs practice focus and fitness to perform well",
    # Health
    "exercise and a balanced diet support good health",
    "the doctor examines a patient at the hospital",
    "protein helps muscle recovery after strength training",
    "fitness improves when people sleep well and exercise regularly",
    "the nurse gives medicine according to the doctor's advice",
    "healthy food provides energy vitamins and minerals",
    "a patient visits the clinic for a medical checkup",
    "strength training can build muscle and improve endurance",
    "health professionals explain treatment and recovery plans",
    "water and nutritious meals support everyday fitness",
    "regular exercise can reduce stress and support wellness",
    "medical research studies disease prevention and patient care",
    # Finance and bank meaning 1
    "the customer deposited money into a bank account",
    "a bank offers savings accounts loans and financial services",
    "the banker checked the account balance and interest rate",
    "people save money in a bank for future expenses",
    "the business applied for a loan from the local bank",
    "online banking lets customers transfer money securely",
    "a financial adviser explains credit savings and investment",
    "the customer withdrew cash from the bank yesterday",
    # Nature and bank meaning 2
    "the river bank was covered with grass and small trees",
    "children sat on the bank of the river near the water",
    "the river flows past a forest and a green valley",
    "trees and plants grow in healthy soil near the river",
    "the mountain stream carries fresh water through the forest",
    "birds and animals live in habitats across the natural world",
    "the river bank becomes muddy after heavy rain",
    "nature includes rivers mountains forests plants and animals",
    # Education
    "students attend college classes and complete assignments",
    "the teacher explains a concept before the examination",
    "learning new skills helps students prepare for their careers",
    "the university library provides books and study materials",
    "a student writes notes while the professor teaches a lesson",
    "the class uses examples to understand a difficult topic",
    "students practice questions before the final exam",
    "education combines reading discussion practice and assessment",
    # Additional cross-topic context
    "students use python to analyze sports and health data",
    "a fitness tracker records data about exercise and sleep",
    "a bank uses computer systems to protect customer information",
    "machine learning can help a hospital organize medical records",
    "a student learns finance by reading examples and solving problems",
    "the coach uses statistics to analyze team performance",
    "scientists use data to study rivers and climate patterns",
    "software helps a college organize student records",
]


def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())

sentences = [tokenize(text) for text in corpus]
print(f"Number of documents: {len(corpus)}")
print(f"Total tokens: {sum(len(s) for s in sentences)}")
print(f"Unique tokens: {len(set(w for s in sentences for w in s))}")
print("Example:", corpus[0])


# %%
tfidf = TfidfVectorizer(ngram_range=(1, 1), lowercase=True, stop_words="english")
X_tfidf = tfidf.fit_transform(corpus)
terms = tfidf.get_feature_names_out()
# A word vector is its TF-IDF weight across documents.
term_vectors = X_tfidf.T
term_to_idx = {term: i for i, term in enumerate(terms)}
print("TF-IDF document matrix shape:", X_tfidf.shape)
print("Vocabulary size:", len(terms))

def most_similar_tfidf(query, topn=5):
    query = query.lower()
    if query not in term_to_idx:
        print(f"'{query}' is not in the TF-IDF vocabulary.")
        return []
    sims = cosine_similarity(term_vectors[term_to_idx[query]], term_vectors).ravel()
    sims[term_to_idx[query]] = -1
    best = np.argsort(sims)[::-1][:topn]
    result = [(terms[i], float(sims[i])) for i in best]
    print(f"TF-IDF words most similar to '{query}':")
    for word, score in result:
        print(f"  {word:18s} cosine={score:.3f}")
    return result

_ = most_similar_tfidf("computer")


# %%
word2vec_model = Word2Vec(
    sentences=sentences,
    vector_size=64,
    window=4,
    min_count=1,
    workers=2,
    sg=0,            # CBOW
    negative=5,
    epochs=100,
    seed=SEED,
)
print("Word2Vec vocabulary size:", len(word2vec_model.wv))

def show_similar(model, query, label, topn=5):
    if query not in model.wv.key_to_index:
        print(f"{query!r} is not in the {label} vocabulary.")
        return []
    results = model.wv.most_similar(query, topn=topn)
    print(f"{label}: five words most similar to '{query}'")
    for word, score in results:
        print(f"  {word:18s} cosine={score:.3f}")
    return results

_ = show_similar(word2vec_model, "computer", "Word2Vec CBOW")
_ = show_similar(word2vec_model, "basketball", "Word2Vec CBOW")


# %%
fasttext_model = FastText(
    sentences=sentences,
    vector_size=64,
    window=4,
    min_count=1,
    workers=2,
    sg=0,
    min_n=2,
    max_n=5,
    epochs=100,
    seed=SEED,
)
print("FastText vocabulary size:", len(fasttext_model.wv))
_ = show_similar(fasttext_model, "bank", "FastText")

rare_word = "computerscience"
print("Is the test word in the training vocabulary?", rare_word in fasttext_model.wv.key_to_index)
try:
    rare_vector = fasttext_model.wv[rare_word]
    print("FastText created a vector for the unseen word:", rare_vector.shape)
    print("Vector is finite:", bool(np.isfinite(rare_vector).all()))
except Exception as exc:
    print("FastText could not create a vector in this environment:", repr(exc))

try:
    _ = word2vec_model.wv[rare_word]
    print("Word2Vec returned a vector for the unseen word.")
except KeyError:
    print("Word2Vec cannot return a learned vector for this unseen word because it was not in the vocabulary.")


# %%
keras_tokenizer = Tokenizer(oov_token="<OOV>")
keras_tokenizer.fit_on_texts(corpus)
sequences = keras_tokenizer.texts_to_sequences(corpus)
word_index = keras_tokenizer.word_index
vocab_size = len(word_index) + 1
max_context_len = 8

contexts, targets = [], []
for seq in sequences:
    for i in range(2, len(seq) + 1):
        contexts.append(seq[max(0, i - max_context_len - 1): i - 1])
        targets.append(seq[i - 1])

X_context = pad_sequences(contexts, maxlen=max_context_len, padding="pre", truncating="pre")
y_target = np.asarray(targets, dtype=np.int32)
print("Training examples:", len(X_context), "| vocabulary size:", vocab_size)

keras_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=32, mask_zero=True, name="word_embedding"),
    GlobalAveragePooling1D(),
    Dense(vocab_size, activation="softmax"),
])
keras_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = keras_model.fit(X_context, y_target, epochs=5, batch_size=32, validation_split=0.1, verbose=0)
print("Final training accuracy: %.3f" % history.history["accuracy"][-1])
print("Final validation accuracy: %.3f" % history.history["val_accuracy"][-1])

keras_embeddings = keras_model.get_layer("word_embedding").get_weights()[0]

def most_similar_keras(query, topn=5):
    q = query.lower()
    if q not in word_index:
        print(f"'{q}' is not in Keras vocabulary")
        return []
    qv = keras_embeddings[word_index[q]]
    # Cosine similarity; skip padding row 0 and the query itself.
    norms = np.linalg.norm(keras_embeddings, axis=1) + 1e-12
    sims = keras_embeddings @ qv / (norms * (np.linalg.norm(qv) + 1e-12))
    sims[0] = -1
    sims[word_index[q]] = -1
    id_to_word = {idx: word for word, idx in word_index.items()}
    best = np.argsort(sims)[::-1][:topn]
    result = [(id_to_word.get(int(i), "<UNK>"), float(sims[i])) for i in best]
    print(f"Keras embedding: words closest to '{q}'")
    for word, score in result:
        print(f"  {word:18s} cosine={score:.3f}")
    return result

_ = most_similar_keras("computer")


# %%
bert_available = False
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    BERT_NAME = "bert-base-uncased"
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
    bert_model = AutoModel.from_pretrained(BERT_NAME)
    bert_model.eval()
    bert_available = True
    print("Loaded pretrained model:", BERT_NAME)
except Exception as exc:
    print("BERT could not be loaded. Check internet access and rerun this cell.")
    print("Details:", type(exc).__name__, str(exc)[:300])


# %%
if bert_available:
    def get_contextual_token_vector(sentence, target="bank"):
        encoded = bert_tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = bert_model(**encoded)
        tokens = bert_tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        target_positions = [i for i, tok in enumerate(tokens) if tok.replace("##", "") == target]
        if not target_positions:
            raise ValueError(f"Target token {target!r} was not found in {tokens}")
        # Select the first occurrence of the target token (excluding [CLS]/[SEP]).
        return outputs.last_hidden_state[0, target_positions[0], :].cpu().numpy(), tokens

    finance_sentence = "She deposited money at the bank yesterday."
    river_sentence = "We sat on the bank beside the river."
    finance_vec, finance_tokens = get_contextual_token_vector(finance_sentence)
    river_vec, river_tokens = get_contextual_token_vector(river_sentence)
    contextual_cosine = float(cosine_similarity(finance_vec.reshape(1, -1), river_vec.reshape(1, -1))[0, 0])
    print("Finance context:", finance_sentence)
    print("Tokens:", finance_tokens)
    print("River context:", river_sentence)
    print("Tokens:", river_tokens)
    print("Cosine similarity between the two contextual 'bank' vectors: %.4f" % contextual_cosine)
else:
    print("BERT contextual experiment skipped because the pretrained model was unavailable.")


# %%
query_words = ["computer", "basketball", "bank", "student"]
comparison_records = []
for query in query_words:
    print("\n" + "=" * 60 + "\nQUERY:", query)
    tfidf_results = most_similar_tfidf(query, topn=5)
    w2v_results = show_similar(word2vec_model, query, "Word2Vec CBOW", topn=5)
    ft_results = show_similar(fasttext_model, query, "FastText", topn=5)
    comparison_records.append({
        "Query word": query,
        "TF-IDF neighbours": ", ".join(w for w, _ in tfidf_results),
        "Word2Vec neighbours": ", ".join(w for w, _ in w2v_results),
        "FastText neighbours": ", ".join(w for w, _ in ft_results),
    })

similarity_comparison = pd.DataFrame(comparison_records)
display(similarity_comparison)


# %%
visual_words = [
    "computer", "software", "python", "data", "algorithm", "model",
    "basketball", "player", "coach", "team", "fitness", "exercise",
    "health", "doctor", "bank", "money", "account", "loan",
    "river", "water", "forest", "tree", "student", "college", "teacher",
]
visual_words = [w for w in visual_words if w in word2vec_model.wv.key_to_index]
word_vectors = np.array([word2vec_model.wv[w] for w in visual_words])
coords = PCA(n_components=2, random_state=SEED).fit_transform(word_vectors)

plt.figure(figsize=(12, 8))
plt.scatter(coords[:, 0], coords[:, 1], alpha=0.8)
for i, word in enumerate(visual_words):
    plt.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=9, xytext=(3, 3), textcoords="offset points")
plt.title("PCA visualization of Word2Vec CBOW embeddings")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("word2vec_pca_visualization.png", dpi=180, bbox_inches="tight")
plt.show()
print("Saved: word2vec_pca_visualization.png")


# %%
export_words = list(word2vec_model.wv.index_to_key[:200])
export_vectors = np.array([word2vec_model.wv[w] for w in export_words])
np.savetxt("vectors.tsv", export_vectors, delimiter="\t", fmt="%.6f")
with open("metadata.tsv", "w", encoding="utf-8") as f:
    f.write("word\n")
    for word in export_words:
        f.write(word + "\n")
print("Saved vectors.tsv with shape", export_vectors.shape)
print("Saved metadata.tsv with", len(export_words), "labels")
