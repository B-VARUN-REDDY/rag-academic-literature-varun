import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Determine the directory where the script resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the cleaned data
data_path = os.path.join(script_dir, '..', 'data', 'cleaned_arxiv_metadata.csv')
df = pd.read_csv(data_path)

# Combine title and abstract for embedding
texts = (df['title'] + '. ' + df['abstract']).tolist()

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(texts, show_progress_bar=True)

# Convert to numpy array
embeddings = np.array(embeddings).astype('float32')

# Initialize FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
index.add(embeddings)

# Define the output directory (same as script directory)
output_dir = script_dir

# Save the index
faiss_index_path = os.path.join(output_dir, 'faiss_index.index')
faiss.write_index(index, faiss_index_path)

# Save the DataFrame with an index for reference
df['embedding_index'] = range(len(df))
metadata_path = os.path.join(output_dir, 'metadata_with_index.csv')
df.to_csv(metadata_path, index=False)

print("Embeddings generated and stored successfully.")
