# Evo-Memory "Think, Act, Refine" Design

## Goal
Refactor the current simple "save-on-success" memory system to the full "Think, Act, Refine" framework described in the Evo-Memory paper.

## Core Components

### 1. Think (Retrieval & Planning)
**Current:** Implicitly uses retrieved code in the prompt.
**New:** Explicit Reasoning Step.
- **Action:** Before generating code, the agent must output a **Reasoning Trace**.
- **Prompt Addition:** 
  > "Review the Retrieved Experiences. Explain WHY they are relevant or why you are adapting/ignoring them. Formulate a step-by-step plan."
- **Benefit:** Prevents blind copying of old code and encourages adaptation.

### 2. Act (Execution)
**Current:** Executes code and captures output.
**New:** No major changes needed, but ensure we capture the *result* (Success/Failure) clearly for the Refine stage.

### 3. Refine (Memory Evolution)
**Current:** Appends new experience if user implies "Success". Discards if "Correction".
**New:** Active Memory Management (Add, Prune, Update).

**Trigger:** When a "Success" is confirmed (via explicit feedback or implicit "New Topic").

**The Evolution Process:**
1.  **Input:** 
    - `New Experience` (Query + Code)
    - `Retrieved Memories` (The specific memories that were used to generate this code)
2.  **LLM Decision (The "Evolver"):**
    - We call the LLM with a specific prompt: "Compare the New Experience with the Retrieved Memories."
    - **Decisions:**
        - **ADD:** Is this a novel task? -> Add it.
        - **UPDATE:** Is this a *better* version of a Retrieved Memory? (e.g., fixed a bug, more efficient) -> **Overwrite** the old memory.
        - **PRUNE:** Is a Retrieved Memory now redundant (covered by the new one) or obsolete? -> **Delete** the old memory.
3.  **Execution:** Apply these changes to `experiences.json`.

## Implementation Details

### Data Structure Changes (`experiences.json`)
- **Add UUIDs:** Each memory entry needs a unique ID to allow reliable Updates and Prunes.
- **Schema:**
  ```json
  {
    "id": "uuid-...",
    "query": "...",
    "code": "...",
    "embedding": [...],
    "timestamp": "..."
  }
  ```

### `memory.py` Changes
- `add_experience`: Generate UUID.
- `update_experience(id, new_code, new_query)`: Update entry.
- `delete_experience(id)`: Remove entry.
- `retrieve_experiences`: Return `id` along with content.

### `jarvis.py` Changes
- **Prompt:** Update System Prompt to enforce "Reasoning Trace".
- **Feedback Loop:** 
  - Inside the `[Evo-Memory]` block (where we currently just `add_experience`):
  - Call a new internal function `evolve_memory_bank(new_exp, retrieved_exps)`.
  - This function performs the LLM comparison and calls `memory` functions.

## Workflow Example
1. **User:** "Plot a sine wave."
2. **Think:** Jarvis retrieves a "Plot graph" memory. Plans: "Use matplotlib, similar to memory #1 but specifically for sine."
3. **Act:** Generates code. User sees it.
4. **User:** "Great, now save it to a file." (Implies previous was good).
5. **Refine:** 
   - Jarvis sees "Success" for "Plot sine wave".
   - **Evolver:** Checks "Plot graph" memory.
   - **Decision:** "The new code is specific to sine, the old one was generic. Keep both (ADD)." OR "The old one was broken, this fixes it (UPDATE)."
