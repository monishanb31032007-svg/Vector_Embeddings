# Week 11 – Vector Embeddings with ChromaDB

## Project Title
**Creating and Storing Vector Embeddings using ChromaDB**

## 1. Introduction

Large Language Models (LLMs) have limitations when answering questions based on custom documents. Vector databases help overcome this limitation by storing document information as numerical vector embeddings.

This project uses the Sentence Transformers library to convert text documents into vector embeddings and stores them in ChromaDB. It performs semantic similarity searches to retrieve the most relevant documents based on the meaning of user queries.

## 2. Objectives

- To understand the role of vector databases in LLM applications.
- To generate vector embeddings using Sentence Transformers.
- To store document embeddings in ChromaDB.
- To perform semantic similarity searches.
- To retrieve the Top-5 most relevant documents for each query.
- To compare semantic search with keyword-based search.
- To prepare a knowledge base for Retrieval-Augmented Generation (RAG).

## 3. Technologies Used

- **Python** – Programming language.
- **Sentence Transformers** – Generates text embeddings.
- **all-MiniLM-L6-v2** – Pre-trained embedding model.
- **ChromaDB** – Vector database.
- **NumPy** – Stores vector embeddings.
- **Pandas** – Handles CSV files and retrieval results.
- **Scikit-learn** – Performs keyword-based TF-IDF search.
- **Jupyter Notebook** – Interactive development environment.
- **VS Code** – Code editor.

## 4. Dataset Description

The project uses Artificial Intelligence as its document domain.

A total of 10 text documents are created:

1. Introduction to Artificial Intelligence
2. Machine Learning
3. Deep Learning
4. Natural Language Processing
5. Generative Artificial Intelligence
6. Computer Vision
7. Artificial Intelligence in Robotics
8. Artificial Intelligence in Healthcare
9. Artificial Intelligence in Education
10. Artificial Intelligence Ethics

Each document contains metadata such as title, category, and source.

A CSV file named `queries.csv` contains 10 sample user queries for testing semantic search.

## 5. Project Folder Structure

```text
Vector_Embeddings/
│
├── dataset/
│   ├── documents/
│   │   ├── document1.txt
│   │   ├── document2.txt
│   │   ├── document3.txt
│   │   ├── document4.txt
│   │   ├── document5.txt
│   │   ├── document6.txt
│   │   ├── document7.txt
│   │   ├── document8.txt
│   │   ├── document9.txt
│   │   └── document10.txt
│   │
│   ├── queries.csv
│   ├── embeddings.npy
│   ├── retrieved_results.csv
│   ├── keyword_comparison.csv
│   └── search_comparison.csv
│
├── chroma_db/
│
├── notebook/
│   └── Week11_Vector_Embeddings.ipynb
│
├── vector_store.py
├── keyword_search.py
├── README.md
└── requirements.txt
```

## 6. Implementation Steps

### Step 1: Install Required Libraries

The required Python libraries are installed using pip.

```bash
pip install -r requirements.txt
```

### Step 2: Load Documents

The Python program reads all text files from the dataset/documents folder.

### Step 3: Generate Vector Embeddings

The Sentence Transformer model `all-MiniLM-L6-v2` converts each document into a numerical vector representation.

Each embedding contains 384 dimensions.

### Step 4: Create ChromaDB Collection

A persistent ChromaDB database is created with a collection named `ai_documents`.

### Step 5: Store