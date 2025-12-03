import json
import os
import google.generativeai as genai
import numpy as np
from datetime import datetime

# Configuration
MEMORY_FILE = os.path.expanduser("memory.json")
EXPERIENCES_FILE = os.path.expanduser("experiences.json")
EMBEDDING_MODEL = "models/text-embedding-004"

def _load_json(filepath):
    if not os.path.exists(filepath):
        return [] if filepath == EXPERIENCES_FILE else {}
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return [] if filepath == EXPERIENCES_FILE else {}

def _save_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

# --- Legacy Key-Value Memory ---

def remember(key, value):
    """Stores a value in memory associated with the given key."""
    memory = _load_json(MEMORY_FILE)
    memory[key] = value
    _save_json(MEMORY_FILE, memory)
    print(f"Remembered: {key} = {value}")

def recall(key):
    """Retrieves a value from memory by its key."""
    memory = _load_json(MEMORY_FILE)
    value = memory.get(key)
    if value:
        print(f"Recalled: {key} = {value}")
    return value

def forget(key):
    """Removes a value from memory by its key."""
    memory = _load_json(MEMORY_FILE)
    if key in memory:
        del memory[key]
        _save_json(MEMORY_FILE, memory)
        print(f"Forgot: {key}")

def list_memories():
    """Returns a list of all keys currently in memory."""
    memory = _load_json(MEMORY_FILE)
    keys = list(memory.keys())
    print(f"Current memories: {keys}")
    return keys

# --- Evo-Memory (Experience Reuse) ---

def get_embedding(text):
    """Generates an embedding for the given text using Gemini."""
    try:
        # Ensure API key is set (it should be if jarvis.py is running)
        if not os.getenv("GEMINI_API_KEY"):
            # Fallback or silent fail if running isolated
            return None
        
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document",
            title="Jarvis Experience"
        )
        return result['embedding']
    except Exception as e:
        print(f"[Memory Error] Failed to generate embedding: {e}")
        return None

def add_experience(query, code, outcome="success"):
    """
    Saves a coding experience.
    query: The user's request
    code: The successful code generated
    outcome: Metadata about success
    """
    experiences = _load_json(EXPERIENCES_FILE)
    
    embedding = get_embedding(query)
    if not embedding:
        print("[Memory] Skipping experience save due to embedding failure.")
        return

    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "code": code,
        "outcome": outcome,
        "embedding": embedding
    }
    
    experiences.append(entry)
    _save_json(EXPERIENCES_FILE, experiences)
    print(f"[Evo-Memory] Experience saved for: '{query}'")

def retrieve_experiences(query, k=3):
    """
    Retrieves the top k most similar experiences for the given query.
    Returns a list of dicts: {'query': ..., 'code': ..., 'score': ...}
    """
    experiences = _load_json(EXPERIENCES_FILE)
    if not experiences:
        return []

    query_embedding = get_embedding(query)
    if not query_embedding:
        return []

    query_vec = np.array(query_embedding)
    
    scored_experiences = []
    for exp in experiences:
        if 'embedding' not in exp:
            continue
        
        exp_vec = np.array(exp['embedding'])
        
        # Cosine similarity
        similarity = np.dot(query_vec, exp_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(exp_vec))
        
        scored_experiences.append({
            "query": exp['query'],
            "code": exp['code'],
            "score": float(similarity)
        })
    
    # Sort by score descending
    scored_experiences.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top k
    top_k = scored_experiences[:k]
    if top_k:
        print(f"[Evo-Memory] Retrieved {len(top_k)} relevant experiences.")
    return top_k

if __name__ == "__main__":
    # Test
    # remember("foo", "bar")
    # add_experience("print hello world", "print('Hello World')")
    pass
