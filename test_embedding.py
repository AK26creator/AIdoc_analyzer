from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

model = SentenceTransformer(MODEL_NAME)

sentences = [
    "What is artificial intelligence?",
    "AI is the simulation of human intelligence by machines.",
    "The weather is hot today."
]

embeddings = model.encode(
    sentences,
    normalize_embeddings=True
)

print("Embedding shape:", embeddings.shape)
print("First embedding:")
print(embeddings[0])