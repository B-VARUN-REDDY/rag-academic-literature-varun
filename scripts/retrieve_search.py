import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Dynamically get current script location
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load FAISS index
index = faiss.read_index(os.path.join(script_dir, 'faiss_index.index'))

# Load metadata
df = pd.read_csv(os.path.join(script_dir, 'metadata_with_index.csv'))

# Load the same embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')


# FUNCTION to search
def search_query(user_query, top_k=5):
    query_embed = model.encode([user_query])
    query_embed = np.array(query_embed).astype('float32')

    distances, indices = index.search(query_embed, top_k)

    results = df.iloc[indices[0]]
    return results


# ----------------------
# Example Usage
query = input("Enter your search query: ")

results = search_query(query)

print("\nTop Results:\n")
for idx, row in results.iterrows():
    print(f"Title   : {row['title']}")
    print(f"Abstract: {row['abstract'][:300]}...")  # Show only first 300 chars
    print(f"Categories: {row['categories']}")
    print(f"Date: {row['update_date']}")
    print("-"*80)
