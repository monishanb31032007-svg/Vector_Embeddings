# ==========================================
# WEEK 11 - KEYWORD BASED SEARCH
# ==========================================

import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Define paths

DOCUMENT_FOLDER = "dataset/documents"
QUERY_FILE = "dataset/queries.csv"
SEMANTIC_FILE = "dataset/retrieved_results.csv"
OUTPUT_FILE = "dataset/keyword_comparison.csv"

# Step 2: Load documents

documents = []
document_ids = []
titles = []

for filename in sorted(os.listdir(DOCUMENT_FOLDER)):

    if filename.endswith(".txt"):

        path = os.path.join(DOCUMENT_FOLDER, filename)

        with open(path, "r", encoding="utf-8") as file:
            content = file.read()

        title = filename.replace(".txt", "")

        for line in content.splitlines():
            if line.startswith("Title:"):
                title = line.replace("Title:", "").strip()

        documents.append(content)
        document_ids.append(filename)
        titles.append(title)

print("Documents loaded:", len(documents))

# Step 3: Create TF-IDF vectors

vectorizer = TfidfVectorizer(
    stop_words="english"
)

document_vectors = vectorizer.fit_transform(documents)

# Step 4: Load queries

queries_df = pd.read_csv(QUERY_FILE)

results = []

# Step 5: Perform keyword search

for _, row in queries_df.iterrows():

    query_id = row["query_id"]
    query = str(row["query"])

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    top_indices = scores.argsort()[::-1][:5]

    for rank, index in enumerate(top_indices, start=1):

        results.append({
            "query_id": query_id,
            "query": query,
            "rank": rank,
            "document_id": document_ids[index],
            "title": titles[index],
            "keyword_score": round(float(scores[index]), 4)
        })

# Step 6: Save keyword search results

keyword_df = pd.DataFrame(results)

keyword_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nKeyword search completed!")

print("Total queries:", len(queries_df))
print("Total results:", len(keyword_df))

print("Results saved to:", OUTPUT_FILE)

# Step 7: Compare semantic and keyword search

semantic_df = pd.read_csv(SEMANTIC_FILE)

semantic_top = semantic_df[
    semantic_df["rank"] == 1
][["query_id", "title"]].copy()

keyword_top = keyword_df[
    keyword_df["rank"] == 1
][["query_id", "title"]].copy()

comparison = semantic_top.merge(
    keyword_top,
    on="query_id",
    suffixes=("_semantic", "_keyword")
)

comparison["same_top_result"] = (
    comparison["title_semantic"] ==
    comparison["title_keyword"]
)

comparison.to_csv(
    "dataset/search_comparison.csv",
    index=False
)

print("\n========== SEARCH COMPARISON ==========")

print(comparison.to_string(index=False))

print(
    "\nSame top result:",
    comparison["same_top_result"].sum(),
    "out of",
    len(comparison)
)

print("\nAll keyword search tasks completed!")