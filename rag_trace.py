from sentence_transformers import SentenceTransformer, util
import chromadb

# --- Step 0: Set up the knowledge base ---
print("=" * 60)
print("STEP 0: Setting up the knowledge base")
print("=" * 60)

client = chromadb.Client()  # In-memory for this demo
collection = client.get_or_create_collection(
    "course_docs", metadata={"hnsw:space": "cosine"}
)

documents = [
    "FastAPI uses Pydantic models for automatic request validation. "
    "Define a Pydantic class with field types and FastAPI validates "
    "incoming data automatically, returning 422 errors for invalid requests.",

    "JWT (JSON Web Token) authentication in FastAPI works by creating "
    "a /auth/token endpoint that validates credentials and returns a "
    "signed token. Protected endpoints verify the token on each request.",

    "Streamlit session state persists data across re-runs using "
    "st.session_state. Initialize with: if 'key' not in st.session_state: "
    "st.session_state['key'] = default_value. Without this, all variables "
    "reset on every widget interaction.",

    "ChromaDB is a vector database that stores document embeddings for "
    "fast similarity search. It supports metadata filtering, persistent "
    "storage, and automatic embedding generation.",

    "CSS Flexbox arranges child elements in a row or column. Apply "
    "display: flex to the container, use gap for spacing, and "
    "flex-wrap: wrap for responsive layouts.",

    "Docker containers package an application with all its dependencies "
    "into a standardized unit. This ensures the application runs the "
    "same way on every machine, solving the 'works on my machine' problem.",
]

sources = [
    "module5_validation.md", "module5_auth.md", "module6_streamlit.md",
    "module7_chromadb.md", "module6_css.md", "module8_docker.md"
]

collection.add(
    documents=documents,
    metadatas=[{"source": s} for s in sources],
    ids=[f"doc_{i}" for i in range(len(documents))]
)
print(f"Knowledge base loaded: {collection.count()} document chunks\n")

# --- Step 1: User asks a question ---
user_question = "How do I protect my API endpoints so only logged-in users can access them?"
print("=" * 60)
print("STEP 1: User Question")
print("=" * 60)
print(f"'{user_question}'\n")

# --- Step 2: Retrieve relevant documents ---
print("=" * 60)
print("STEP 2: Retrieve from ChromaDB")
print("=" * 60)

results = collection.query(
    query_texts=[user_question],
    n_results=3
)

print(f"Found {len(results['documents'][0])} relevant chunks:\n")
for i in range(len(results['documents'][0])):
    doc = results['documents'][0][i]
    source = results['metadatas'][0][i]['source']
    distance = results['distances'][0][i]
    print(f"  [{i+1}] (dist: {distance:.4f}) [{source}]")
    print(f"      {doc[:80]}...\n")

# --- Step 3: Build the prompt ---
print("=" * 60)
print("STEP 3: Build the Prompt")
print("=" * 60)

context = "\n\n".join([
    f"[Source: {results['metadatas'][0][i]['source']}]\n{results['documents'][0][i]}"
    for i in range(len(results['documents'][0]))
])

system_prompt = (
    "You are a helpful AI assistant for students learning AI engineering. "
    "Answer the user's question based ONLY on the context provided below. "
    "If the context doesn't contain enough information to answer, say so. "
    "Always cite which source document your answer comes from. "
    "Keep your response under 150 words."
)

full_prompt = f"""SYSTEM: {system_prompt}

CONTEXT:
{context}

USER QUESTION: {user_question}

ANSWER:"""

print(f"Prompt length: {len(full_prompt)} characters")
print(f"\n--- Full Prompt (what gets sent to the LLM) ---")
print(full_prompt[:500] + "\n...\n")

# --- Step 4: Generate (mock for this demo) ---
print("=" * 60)
print("STEP 4: LLM Generation (mock response)")
print("=" * 60)

mock_answer = (
    "To protect your API endpoints, use JWT authentication. Create a "
    "/auth/token endpoint that validates user credentials and returns a "
    "signed token. Then, add a dependency to your protected endpoints that "
    "verifies the token on each request. If the token is missing or invalid, "
    "FastAPI returns a 401 Unauthorized response.\n\n"
    "Source: module5_auth.md"
)

print(f"\n{mock_answer}")

print("\n" + "=" * 60)
print("SUMMARY: The RAG pipeline")
print("=" * 60)
print("1. User asked about protecting API endpoints")
print("2. ChromaDB found the JWT authentication document (most relevant)")
print("3. We built a prompt with context + system instructions")
print("4. The LLM generated an answer grounded in the retrieved docs")
print("5. The answer cites its source (module5_auth.md)")
print("\nIn the next lessons, you'll replace the mock with a real LLM (Ollama)!")
