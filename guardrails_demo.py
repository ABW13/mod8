"""Demo: retrieval guardrails, with and without.

Compares a naive pipeline that sends every retrieved chunk to the model against
a guarded one that filters by distance, reports a confidence level, and refuses
when nothing is close enough.
"""

import chromadb
from chromadb.utils import embedding_functions

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB defaults to L2 distance; the thresholds in this course assume
# cosine, so the space is set explicitly. Without this the distances below
# would be roughly double their cosine equivalents and every cutoff wrong.
COSINE_SPACE = {"hnsw:space": "cosine"}

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

# --- Setup: mini knowledge base ---
client = chromadb.Client()
collection = client.get_or_create_collection(
    "demo",
    embedding_function=embedder,
    metadata=COSINE_SPACE,
)

documents = [
    "FastAPI uses Pydantic models for automatic request validation.",
    "JWT tokens provide stateless authentication for REST APIs.",
    "ChromaDB stores document embeddings for similarity search.",
    "Streamlit session state persists data across re-runs.",
]
collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))],
    metadatas=[{"source": f"module{i + 3}.md"} for i in range(len(documents))],
)


# --- Without guardrails ---


def naive_retrieve(query, n_results=3):
    """Return the top-k chunks with no relevance filtering at all."""
    results = collection.query(query_texts=[query], n_results=n_results)
    return [
        {
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i],
            "source": results["metadatas"][0][i]["source"],
        }
        for i in range(len(results["documents"][0]))
    ]


# --- With guardrails ---


def safe_retrieve(query, n_results=3, max_distance=0.75):
    """Return only chunks within max_distance.

    The cutoff is what separates "related" from "merely nearest". ChromaDB
    always returns n_results, so without a filter an out-of-scope query still
    yields its three nearest chunks — and they go to the model.
    """
    results = collection.query(query_texts=[query], n_results=n_results)
    filtered = []
    for i in range(len(results["documents"][0])):
        dist = results["distances"][0][i]
        if dist <= max_distance:
            filtered.append({
                "text": results["documents"][0][i],
                "distance": dist,
                "source": results["metadatas"][0][i]["source"],
            })
    return filtered


def get_confidence(chunks):
    """Map the best distance to a confidence level.

    Thresholds are in cosine distance: lower is better.
      0.00 - 0.50  high
      0.50 - 0.75  medium
      0.75 +       low
    """
    if not chunks:
        return "none", "RED"
    best = chunks[0]["distance"]
    if best < 0.5:
        return "high", "GREEN"
    if best < 0.75:
        return "medium", "YELLOW"
    return "low", "RED"


def answer_with_guardrails(query):
    """Retrieve with filtering and report what the pipeline would do."""
    chunks = safe_retrieve(query, max_distance=0.75)
    confidence, icon = get_confidence(chunks)

    print(f"\n  Confidence: [{icon}] {confidence}")
    print(f"  Chunks retrieved: {len(chunks)}")

    if not chunks:
        print("  Answer: I don't have relevant information to answer that.")
        return

    for c in chunks:
        print(f"    [{c['distance']:.3f}] {c['source']}: {c['text'][:60]}...")

    if confidence == "low":
        print("  Answer: I found some loosely related information, but I'm "
              "not confident it answers your question:")
    print(f"  Answer: [Would generate from {len(chunks)} chunks using Ollama]")


# --- Compare ---

queries = [
    "How does FastAPI validate data?",   # in scope, should answer well
    "How do I deploy to Kubernetes?",    # out of scope, should refuse
    "What is a vector database?",        # partially in scope
]

for query in queries:
    print(f"\n{'=' * 60}")
    print(f"Query: {query!r}")

    print("\n  --- Without guardrails ---")
    naive = naive_retrieve(query)
    for c in naive:
        print(f"    [{c['distance']:.3f}] {c['source']}: {c['text'][:60]}...")
    best = naive[0]["distance"] if naive else None
    print(f"  [Would send ALL {len(naive)} chunks to the LLM, "
          f"best distance {best:.3f}]")

    print("\n  --- With guardrails ---")
    answer_with_guardrails(query)
