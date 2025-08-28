import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Paths
PRODUCTS_JSON = "Model/merged_data_for_poc.json"
CHROMA_PERSIST_DIR = "chroma_persist"

# Load products
with open(PRODUCTS_JSON, "r", encoding="utf-8") as f:
    products = json.load(f)

# Initialize encoder
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Prepare ChromaDB persistent client
client = chromadb.Client(Settings(
    persist_directory=CHROMA_PERSIST_DIR
))
collection = client.get_or_create_collection("products")

# Create and store embeddings
for prod in products:
    product_number = prod.get("productNumber")
    features = prod.get("productFeatures", [])
    # Create a string representation of features
    feature_str = " ".join([f"{f['featureId']}:{f['value']}" for f in features])
    # Create embedding
    embedding = encoder.encode(feature_str)
    # Store in ChromaDB
    collection.add(
        documents=[feature_str],
        embeddings=[embedding.tolist()],
        ids=[str(product_number)]
    )

# Persist the database
print("Embeddings stored in ChromaDB.")