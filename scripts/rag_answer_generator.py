import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import openai

# Initialize OpenAI client with API key
client = openai.OpenAI(api_key="YOUR_API_KEY")

script_dir = os.path.dirname(os.path.abspath(__file__))

index = faiss.read_index(os.path.join(script_dir, 'faiss_index.index'))
df = pd.read_csv(os.path.join(script_dir, 'metadata_with_index.csv'))
model = SentenceTransformer('all-MiniLM-L6-v2')


def search_query(user_query, top_k=5):
    query_embed = model.encode([user_query])
    query_embed = np.array(query_embed).astype('float32')
    distances, indices = index.search(query_embed, top_k)
    return df.iloc[indices[0]]


def generate_rag_answer(query):
    results = search_query(query, top_k=5)
    context = "\n\n".join(results['abstract'].tolist())

    prompt = f"""
You are an expert assistant.

Use the below context from research papers to answer the question clearly and accurately.

Context:
{context}

Question: {query}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You provide research-backed, clear, structured answers."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500,
    )

    print("\nGenerated Answer:\n")
    print(response.choices[0].message.content)

    print("\nSources Used:\n")
    for idx, row in results.iterrows():
        print(f"- {row['title']} ({row['update_date']})")


# -------------------------
query = input("Ask your Question: ")
generate_rag_answer(query)
