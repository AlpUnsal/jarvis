import json
import os

MEMORY_FILE = os.path.expanduser("memory.json")

def _load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def _save_memory(memory):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=4)

def remember(key, value):
    """Stores a value in memory associated with the given key."""
    memory = _load_memory()
    memory[key] = value
    _save_memory(memory)
    print(f"Remembered: {key} = {value}")

def recall(key):
    """Retrieves a value from memory by its key. Returns None if not found."""
    memory = _load_memory()
    value = memory.get(key)
    if value:
        print(f"Recalled: {key} = {value}")
    else:
        print(f"Memory not found for key: {key}")
    return value

def forget(key):
    """Removes a value from memory by its key."""
    memory = _load_memory()
    if key in memory:
        del memory[key]
        _save_memory(memory)
        print(f"Forgot: {key}")
    else:
        print(f"Key not found in memory: {key}")

def list_memories():
    """Returns a list of all keys currently in memory."""
    memory = _load_memory()
    keys = list(memory.keys())
    print(f"Current memories: {keys}")
    return keys

if __name__ == "__main__":
    # Simple test
    remember("test_key", "test_value")
    recall("test_key")
    list_memories()
    forget("test_key")
