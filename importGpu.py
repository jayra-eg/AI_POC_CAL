from chromadb import PersistentClient
client = PersistentClient(path="./chroma_persist")
col = client.get_collection("products_embeddings")
sample = col.get(limit=1, include=["embeddings", "documents"])
print(sample)
