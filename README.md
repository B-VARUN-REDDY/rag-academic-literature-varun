RAG System for Academic Literature (ScriptChain Health — Take Home Assignment)
Overview
A Retrieval-Augmented Generation (RAG) system designed to help researchers query academic literature (arxiv data) and get AI-generated answers with proper citations from relevant papers.

Tech Stack Used
Sentence Transformers → For generating embeddings

FAISS → For fast vector similarity search

GPT4All (Mistral 7B) → Local free LLM for generating answers

Python (Command Line Interface)

Project Structure

RAG_Project/
│
├── scripts/                        # All Python scripts
│   ├── rag_answer_generator_local.py
│   ├── preprocess_arxiv_data.py
│   ├── generate_embeddings.py
│   └── load_arxiv_data.py
│
├── data/                           # Processed data & FAISS index
│   ├── faiss_index.index
│   ├── metadata_with_index.csv
│
├── models/                         # Local LLM model
│   └── mistral-7b-openorca.Q4_0.gguf
│
├── requirements.txt                # Python dependencies
├── evaluation_results.txt          # Evaluation Logs
└── README.md
Setup Instructions (Local)
bash
Copy
Edit
# Clone the Repo
git clone https://github.com/B-VARUN-REDDY/rag-academic-literature-varun.git

cd RAG_Project

# Create Virtual Environment
python -m venv venv

# Activate Environment
venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt

# Run the RAG System
python scripts/rag_answer_generator_local.py
Usage Example
→ Simply run and input your question:

python
Copy
Edit
Ask your Question: evolution of earth moon system
Output Includes:
- ✅ AI Generated Answer  
- 📚 Sources Used (Proper Citations)  
- 📊 Similarity Scores  
- ⏱️ Time Taken  

Logs stored in evaluation_results.txt

Video Demo
🎥 Watch Demo Video Here → https://drive.google.com/file/d/1KZiZKiw_vEVEj7J28eU4wxYezP3SeSlT/view?usp=sharing

Dataset Download
Since arxiv dataset is >300MB (GitHub Limit), download it here:

📥 [Download arxiv-metadata-oai-snapshot.json from Releases](https://github.com/B-VARUN-REDDY/rag-academic-literature-varun/releases)

1. Download `arxiv-metadata-oai-snapshot.json`  
2. Place it inside → `/data` folder

## Architecture Writeup
This RAG pipeline uses:
- FAISS to store sentence-transformer-based vector embeddings
- A top-K semantic retriever based on cosine similarity
- A local Mistral-7B model via GPT4All to generate answers
- Evaluation logged with similarity scores and user feedback


Author
Varun Reddy Bhimavarapu
Email: varunbhimavarapu007@gmail.com
GitHub: B-VARUN-REDDY