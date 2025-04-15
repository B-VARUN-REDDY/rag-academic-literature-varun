# RAG System for Academic Literature (ScriptChain Health Assignment)

## Overview
A Retrieval-Augmented Generation (RAG) system that allows users to query academic literature (arxiv data) and get answers backed by citations.

Built using:
- Sentence Transformers (Embeddings)
- FAISS (Vector Search)
- GPT4All (Local LLM)
- Mistral 7B Model (Offline)

---

## Folder Structure

```
RAG_Project/
│
├── scripts/
│   ├── rag_answer_generator_local.py
│   ├── preprocess_arxiv_data.py
│   └── generate_embeddings.py
│
├── data/
│   ├── faiss_index.index
│   ├── metadata_with_index.csv
│
├── models/
│   └── mistral-7b-openorca.Q4_0.gguf
│
├── requirements.txt
├── evaluation_results.txt
└── README.md
```

---

## Setup Instructions

```bash
git clone <your-repo-link>
cd RAG_Project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts/rag_answer_generator_local.py
```

---

## Usage
Run → Ask Questions → Get Answers with Sources.

Example:

```
Ask your Question: evolution of earth moon system
```

---

## Output
- Generated Answer
- Sources used
- Evaluation Results stored in → `evaluation_results.txt`

---

## Video Demo
[Upload your 5-min video to Google Drive/Youtube & paste link here]

---

## Author
Varun Reddy Bhimavarapu

