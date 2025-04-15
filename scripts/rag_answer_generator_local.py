import pandas as pd
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import os
import time
from gpt4all import GPT4All

script_dir = os.path.dirname(os.path.abspath(__file__))


# --------------------- Auto-create requirements.txt ---------------------
req_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')

required_packages = [
    "faiss-cpu",
    "pandas",
    "numpy",
    "sentence-transformers",
    "gpt4all",
    "python-dotenv"
]

if not os.path.exists(req_path):
    with open(req_path, 'w') as f:
        f.write('\n'.join(required_packages))
    print("✅ requirements.txt created successfully.")

# --------------------- Load Model and Data ---------------------
model_path = os.path.join(script_dir, '..', 'models', 'mistral-7b-openorca.Q4_0.gguf')

model = GPT4All(model_path)

script_dir = os.path.dirname(os.path.abspath(__file__))

index = faiss.read_index(os.path.join(script_dir, '..', 'data', 'faiss_index.index'))
df = pd.read_csv(os.path.join(script_dir, '..', 'data', 'metadata_with_index.csv'))
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# --------------------- Semantic Search ---------------------
def search_query(user_query, top_k=5):
    query_embed = embed_model.encode([user_query])
    query_embed = np.array(query_embed).astype('float32')
    distances, indices = index.search(query_embed, top_k)
    return df.iloc[indices[0]]

# --------------------- Answer + Evaluation ---------------------
def generate_local_answer(query):
    start = time.time()
    results = search_query(query)

    context = "\n\n".join(results['abstract'].tolist())

    prompt = f"""
Act like a smart research assistant.
Use the below context from academic papers to answer clearly.

Context:
{context}

Question:
{query}

Answer:
"""

    with model.chat_session() as chat:
        response = chat.generate(prompt, max_tokens=500)

    print("\nGenerated Answer:\n")
    print(response)

    print("\nSources Used:\n")
    for idx, row in results.iterrows():
        print(f"- {row['title']} ({row['update_date']})")

    # ------------------ Evaluation ------------------
    feedback = input("\nWas this answer useful? (yes/no): ")

    query_embedding = embed_model.encode([query], convert_to_tensor=True)
    result_embeddings = embed_model.encode(results['abstract'].tolist(), convert_to_tensor=True)
    similarity_scores = util.cos_sim(query_embedding, result_embeddings)

    end = time.time()
    total_time = round(end - start, 2)

    print("\nSimilarity Scores with top results:\n", similarity_scores)
    print(f"\nTime Taken: {total_time} seconds")

    with open("evaluation_results.txt", "a", encoding='utf-8') as f:
        f.write(f"Query: {query}\nFeedback: {feedback}\nSimilarity Scores: {similarity_scores.tolist()}\nTime: {total_time} sec\n\n")

    print("✅ Evaluation results saved to evaluation_results.txt")

# --------------------- Main Run ---------------------
if __name__ == "__main__":
    query = input("Ask your Question: ")
    generate_local_answer(query)
