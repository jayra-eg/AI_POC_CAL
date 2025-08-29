# build_embeddings.py

import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
import torch

# Paths
PRODUCTS_JSON = r"C:\Users\mshar\Desktop\New folder\AI_POC_CAL\data\merged_data_for_poc.json"
CHROMA_PERSIST_DIR = "chroma_persist"

# Load products
with open(PRODUCTS_JSON, "r", encoding="utf-8") as f:
    products = json.load(f)

print(f"Loaded {len(products)} products.")

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Initialize encoder on GPU
encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Prepare ChromaDB persistent client
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = client.get_or_create_collection("products")

# Parameters
BATCH_SIZE = 512   # adjust based on VRAM
INSERT_CHUNK = 1000  # how many embeddings to insert per add() call

# Preprocess: collect product texts, ids, metadata
texts, ids, metas = [], [], []
for prod in products:
    product_number = str(prod.get("productNumber"))
    features = prod.get("productFeatures", [])
    feature_str = " ".join([f"{f['featureId']}:{f['value']}" for f in features])
    if not feature_str.strip():
        continue
    texts.append(feature_str)
    ids.append(product_number)
    metas.append({"productNumber": product_number})

print(f"Prepared {len(texts)} items for embedding.")

# Encode in batches on GPU
embeddings = []
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding embeddings"):
    batch_texts = texts[i:i+BATCH_SIZE]
    batch_embs = encoder.encode(
        batch_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_tensor=False,
        device=device
    )
    embeddings.extend(batch_embs)

print("✅ Finished encoding. Now storing in ChromaDB...")

# Insert into ChromaDB in chunks
for i in tqdm(range(0, len(texts), INSERT_CHUNK), desc="Storing in ChromaDB"):
    chunk_texts = texts[i:i+INSERT_CHUNK]
    chunk_ids = ids[i:i+INSERT_CHUNK]
    chunk_metas = metas[i:i+INSERT_CHUNK]
    chunk_embs = embeddings[i:i+INSERT_CHUNK]

    collection.add(
        documents=chunk_texts,
        embeddings=[e.tolist() for e in chunk_embs],
        ids=chunk_ids,
        metadatas=chunk_metas,
    )

print(f"✅ Done. Embeddings persisted in: {CHROMA_PERSIST_DIR}")
