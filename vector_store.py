# =====================================================
# WEEK 11 - VECTOR EMBEDDINGS WITH CHROMADB
# Project: Creating and Storing Vector Embeddings
# =====================================================

import os
import numpy as np
import pandas as pd
import chromadb

from sentence_transformers import SentenceTransformer


# -----------------------------------------------------
# STEP 1: DEFINE PROJECT PATHS
# -----------------------------------------------------

DOCUMENT_FOLDER = "dataset/documents"
QUERY_FILE = "dataset/queries.csv"
EMBEDDING_FILE = "dataset/embeddings.npy"
RESULT_FILE = "dataset/retrieved_results.csv"
CHROMA_FOLDER = "chroma_db"

os.makedirs(CHROMA_FOLDER, exist_ok=True)


# -----------------------------------------------------
# STEP 2: LOAD SENTENCE TRANSFORMER MODEL
# -----------------------------------------------------

print("\nLoading Sentence Transformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully!")


# -----------------------------------------------------
# STEP 3: LOAD TEXT DOCUMENTS
# -----------------------------------------------------

documents = []
document_ids = []
metadatas = []

print("\nLoading documents...")

for filename in sorted(os.listdir(DOCUMENT_FOLDER)):

    if filename.endswith(".txt"):

        file_path = os.path.join(
            DOCUMENT_FOLDER, filename
        )

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().strip()

        if not content:
            continue

        # Extract title and category
        title = filename.replace(".txt", "")
        category = "Artificial Intelligence"

        for line in content.splitlines():

            if line.startswith("Title:"):
                title = line.replace("Title:", "").strip()

            elif line.startswith("Category:"):
                category = line.replace("Category:", "").strip()

        documents.append(content)
        document_ids.append(filename)

        metadatas.append({
            "title": title,
            "category": category,
            "source": filename
        })

print("Total documents loaded:", len(documents))

if len(documents) == 0:
    raise ValueError("No text documents found!")

if len(documents) < 5:
    raise ValueError("At least 5 documents are required for Top-5 search.")


# -----------------------------------------------------
# STEP 4: GENERATE VECTOR EMBEDDINGS
# -----------------------------------------------------

print("\nGenerating vector embeddings...")

embeddings = model.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Embedding shape:", embeddings.shape)

# Save embeddings in NumPy format
np.save(EMBEDDING_FILE, embeddings)

print("Embeddings saved successfully!")


# -----------------------------------------------------
# STEP 5: CREATE CHROMADB DATABASE
# -----------------------------------------------------

print("\nCreating ChromaDB database...")

client = chromadb.PersistentClient(
    path=CHROMA_FOLDER
)

# Remove the previous collection if it exists
try:
    client.delete_collection("ai_documents")
except Exception:
    pass

# Create collection using cosine distance
collection = client.create_collection(
    name="ai_documents",
    metadata={
        "description": "AI document vector database",
        "hnsw:space": "cosine"
    }
)

print("ChromaDB collection created successfully!")


# -----------------------------------------------------
# STEP 6: STORE DOCUMENTS AND METADATA
# -----------------------------------------------------

print("\nStoring documents in ChromaDB...")

collection.add(
    ids=document_ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

print("Documents stored successfully!")


# -----------------------------------------------------
# STEP 7: LOAD USER QUERIES
# -----------------------------------------------------

queries_df = pd.read_csv(QUERY_FILE)

if not {"query_id", "query"}.issubset(queries_df.columns):
    raise ValueError(
        "queries.csv must contain query_id and query columns."
    )

print("\nTotal user queries:", len(queries_df))


# -----------------------------------------------------
# STEP 8: PERFORM TOP-5 SIMILARITY SEARCH
# -----------------------------------------------------

results = []

print("\nPerforming semantic similarity searches...\n")

for _, row in queries_df.iterrows():

    query_id = row["query_id"]
    query_text = str(row["query"])

    # Generate query embedding
    query_embedding = model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Search for the five most relevant documents
    search_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    # Process retrieved results
    for rank in range(len(search_results["ids"][0])):

        doc_id = search_results["ids"][0][rank]

        doc_text = search_results["documents"][0][rank]

        metadata = search_results["metadatas"][0][rank]

        distance = search_results["distances"][0][rank]

        # Convert cosine distance into cosine similarity
        similarity = 1 - distance

        results.append({
            "query_id": query_id,
            "query": query_text,
            "rank": rank + 1,
            "document_id": doc_id,
            "title": metadata["title"],
            "category": metadata["category"],
            "source": metadata["source"],
            "similarity_score": round(float(similarity), 4),
            "retrieved_document": doc_text
        })

        if rank == 0:
            print("Query:", query_text)
            print("Top Result:", metadata["title"])
            print("Similarity Score:", round(float(similarity), 4))
            print("-" * 50)


# -----------------------------------------------------
# STEP 9: SAVE RETRIEVAL RESULTS TO CSV
# -----------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_FILE,
    index=False
)

print("\nRetrieval results saved successfully!")


# -----------------------------------------------------
# STEP 10: DISPLAY FINAL PROJECT SUMMARY
# -----------------------------------------------------

print("\n========================================")
print("       WEEK 11 PROJECT SUMMARY")
print("========================================")

print("Total documents:", collection.count())

print("Total queries:", len(queries_df))

print("Total retrieved results:", len(results_df))

print("Embedding dimensions:", embeddings.shape[1])

print("\nGenerated Files:")
print("1. dataset/embeddings.npy")
print("2. dataset/retrieved_results.csv")
print("3. chroma_db/")

print("\nALL TASKS COMPLETED SUCCESSFULLY!")

print("========================================")