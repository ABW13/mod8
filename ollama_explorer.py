"""Explore system prompts, context grounding, timing, and temperature."""

import time
import requests

BASE_URL = "http://127.0.0.1:11434"
MODEL = "llama3.2:1b"


def generate(prompt, system=None, temperature=None, timeout=600):
    """Send a prompt to the model and return (text, elapsed_seconds)."""
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {"model": MODEL, "messages": messages, "stream": False}
    if temperature is not None:
        payload["options"] = {"temperature": temperature}

    start = time.time()
    response = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=timeout)
    elapsed = time.time() - start

    if response.status_code != 200:
        return f"[ERROR {response.status_code}] {response.text[:200]}", elapsed

    return response.json()["message"]["content"], elapsed


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


try:
    tags = requests.get(f"{BASE_URL}/api/tags", timeout=5).json()
    models = [m["name"] for m in tags.get("models", [])]
    print(f"Ollama is running. Available models: {models}")
except Exception as exc:
    raise SystemExit(f"Cannot reach Ollama: {exc}")

if not any(m.startswith(MODEL) for m in models):
    raise SystemExit(f"Model {MODEL!r} not found. Run: ollama pull {MODEL}")


header("EXPERIMENT 1: Same question, three system prompts")

for label, system in [
    ("(a) No system prompt", None),
    ("(b) Explain like I'm 5 years old", "Explain like I'm 5 years old."),
    ("(c) Senior software architect",
     "You are a senior software architect. Be technical and precise."),
]:
    text, elapsed = generate("What is an API?", system=system)
    print(f"\n--- {label} ---")
    print(f"({len(text.split())} words, {elapsed:.1f}s)")
    print(text)


header("EXPERIMENT 2: Context grounding (answerable vs not)")

CONTEXT = (
    "Pyramid Coffee Roasters was founded in 2014 in Asheville, North Carolina. "
    "The company sources green beans from smallholder farms in Ethiopia, "
    "Colombia, and Guatemala, and roasts everything in-house at its Asheville "
    "facility. Its best-selling blend is the Blue Ridge Dark, a dark roast "
    "with notes of cocoa and toasted walnut. Pyramid sells wholesale to "
    "restaurants and cafes across the southeastern United States, and operates "
    "a single retail shop next to the roastery."
)

grounded = (
    "Answer the user's question using ONLY the context provided below. "
    "If the context does not contain the answer, reply exactly: "
    "'I don't have enough information to answer that.'\n\n"
    f"CONTEXT:\n{CONTEXT}"
)

for label, question in [
    ("Answerable", "Where does Pyramid Coffee Roasters source its beans?"),
    ("Unanswerable", "What is the company's annual revenue?"),
]:
    text, elapsed = generate(question, system=grounded)
    print(f"\n--- {label}: {question!r} ---")
    print(f"({len(text.split())} words, {elapsed:.1f}s)")
    print(text)
    if label == "Unanswerable":
        refused = "don't have enough information" in text.lower()
        print(f"\n  >> Model {'correctly refused' if refused else 'did NOT refuse'}")


header("EXPERIMENT 3: Response time vs question length")

rows = []
for label, question in [
    ("Short (~5 words)", "What is an API?"),
    ("Medium (~20 words)",
     "What is an API, and how does it differ from a regular function call "
     "inside a program?"),
    ("Long (~50 words)",
     "What is an API, how does it differ from a regular function call inside "
     "a program, what role do HTTP status codes play in a REST API, and why "
     "do teams separate the API contract from the implementation behind it?"),
]:
    text, elapsed = generate(question, system="Be concise. Answer in 2 sentences.")
    rows.append((label, len(question.split()), len(text.split()), elapsed))
    print(f"{label:<20} in={len(question.split()):>3}w  "
          f"out={len(text.split()):>3}w  {elapsed:.2f}s")

print(f"\n  {'question':<20} {'in words':>9} {'out words':>10} {'seconds':>9}")
for label, iw, ow, el in rows:
    print(f"  {label:<20} {iw:>9} {ow:>10} {el:>9.2f}")


header("EXPERIMENT 4: Temperature")

prompt = "Write one sentence describing what an API does."

for temp in [0.1, 1.0]:
    text, _ = generate(prompt, system="You are a creative writer.", temperature=temp)
    print(f"\n--- temperature = {temp} ---")
    print(text)

print("\n  Same prompt again at 1.0:")
second, _ = generate(prompt, system="You are a creative writer.", temperature=1.0)
print(f"  {second}")

header("DONE - record your observations above")
