import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
import subprocess

load_dotenv()

# ---------------- CONFIGURATION ----------------
# Get your free key from: https://openrouter.ai/keys
# For security, it's best to set this as an env var: export OPENROUTER_API_KEY='sk-...'
API_KEY = os.getenv("OPENROUTER_API_KEY")

# OpenRouter Base URL (Standard OpenAI format)
BASE_URL = "https://openrouter.ai/api/v1"

# The specific free coding model ID
# You can swap this for "deepseek/deepseek-r1" or "meta-llama/llama-3.3-70b-instruct:free"
MODEL_ID = "meta-llama/llama-3.3-70b-instruct:free" 
# -----------------------------------------------

def query_coding_model():
    # Initialize the client pointing to OpenRouter instead of OpenAI
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )

    print(f"\n--- Connected to {MODEL_ID} ---")
    print("Type your coding prompt below (type 'quit' to exit).\n")

    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ["quit", "exit"]:
                print("Exiting...")
                break
            
            print("\nThinking...\n")

            # Call the API
            stream = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": "ONLY RESPOND WITH CODE. When the user asks you a question, break down the question into smaller sub-problems. With these sub-problems, define DRY functions to use as building blocks. Then, implement the functions in a clean, efficient way. If the user asks you to open a website, write code to open the website in python. IMPORTANT: When using file paths, ALWAYS use os.path.expanduser('~') to resolve the home directory. NEVER use '~' directly in paths. You are operating the laptop of the user. Every script you write is being run on the user's laptop, it is a macbook pro with the M1 chip. Always run the script in the user's home directory using cwd='~'."},
                    {"role": "user", "content": user_input}
                ],
                stream=True  # Enable streaming for faster feel
            )

            # Print chunks as they arrive and accumulate full response
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    sys.stdout.write(content)
                    sys.stdout.flush()
                    full_response += content
            
            print("\n") # Newline after full response

            # Clean the response (remove markdown code fences)
            clean_code = full_response
            if "```python" in clean_code:
                clean_code = clean_code.split("```python")[1]
                if "```" in clean_code:
                    clean_code = clean_code.split("```")[0]
            elif "```" in clean_code:
                # Fallback if language not specified
                clean_code = clean_code.split("```")[1]
                if "```" in clean_code:
                    clean_code = clean_code.split("```")[0]
            
            clean_code = clean_code.strip()

            # Save to output.py
            with open("output.py", "w") as f:
                f.write(clean_code)
            
            print("--- Saved to output.py ---")
            print("--- Executing output.py ---")
            
            # Execute output.py
            try:
                subprocess.run([sys.executable, "output.py"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error executing code: {e}")
            except Exception as e:
                print(f"Execution failed: {e}")

        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    query_coding_model()
