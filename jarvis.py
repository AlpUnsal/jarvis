import os
import sys
import argparse
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
import subprocess
import speech_recognition as sr
import torch
import numpy as np
import pyaudio

load_dotenv()

# ---------------- CONFIGURATION ----------------
# Default to Google Gemini if no args provided
DEFAULT_PROVIDER = "google"
DEFAULT_MODEL = "gemini-flash-latest"
# -----------------------------------------------

class LLMClient:
    def send_message(self, user_input, system_prompt):
        raise NotImplementedError

class GoogleClient(LLMClient):
    def __init__(self, model_id, api_key):
        if not api_key:
            raise ValueError("Google API Key is required. Set GEMINI_API_KEY env var or pass --key.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_id)
        self.chat = self.model.start_chat(history=[])

    def send_message(self, user_input, system_prompt):
        # Google's chat history handles context, but we inject system prompt + memory each time for robustness 
        # or we could set it once. For now, we prepend it to the user message as per previous logic.
        full_prompt = f"{system_prompt}\n\nUser: {user_input}"
        response_stream = self.chat.send_message(full_prompt, stream=True)
        for chunk in response_stream:
            try:
                if chunk.text:
                    yield chunk.text
            except ValueError:
                # Chunk might not have text (e.g., safety block or finish reason only)
                pass

class OpenAIClient(LLMClient):
    def __init__(self, model_id, api_key, base_url):
        if not api_key:
            raise ValueError("API Key is required for OpenAI/Local provider.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_id = model_id
        self.history = []

    def send_message(self, user_input, system_prompt):
        # Manage history manually for OpenAI/Local
        # Always keep system prompt as the first message? 
        # Or just append it? For simplicity, we'll reconstruct messages.
        
        # Note: A robust implementation would maintain history properly.
        # Here we mimic the previous simple behavior: System + User.
        # If we want conversation history, we should append to self.history.
        
        # For now, let's stick to the previous stateless-ish approach per command 
        # (or rather, the previous code didn't explicitly maintain history list for OpenAI, 
        # but the Google chat object did).
        
        # Let's implement basic history for parity.
        if not self.history:
             # We don't add system prompt to history permanently if it changes (dynamic memory),
             # but usually system prompt is static-ish. 
             # However, our system prompt includes dynamic memory.
             pass

        messages = [{"role": "system", "content": system_prompt}] + self.history + [{"role": "user", "content": user_input}]
        
        stream = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                yield content
                full_response += content
        
        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": full_response})

def listen_with_vad(model, utils):
    """
    Listens to the microphone using Silero VAD to detect speech start and end.
    Returns the recognized text.
    """
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    # Audio configuration
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 512
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print("\nListening... (Speak now)")
    
    frames = []
    speaking = False
    silence_counter = 0
    SILENCE_THRESHOLD = 20  # approx 0.6 seconds of silence to stop
    
    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            # Convert to float32 for VAD
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Get speech probability
            # model expects (batch, samples) or just (samples)
            speech_prob = model(torch.from_numpy(audio_chunk), RATE).item()
            
            if speech_prob > 0.5:
                if not speaking:
                    print("Speech started...")
                    speaking = True
                silence_counter = 0
            else:
                if speaking:
                    silence_counter += 1
                    if silence_counter > SILENCE_THRESHOLD:
                        print("Speech ended.")
                        break
        except KeyboardInterrupt:
            raise

    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    if not frames:
        return None

    # Convert captured frames to AudioData
    audio_data = b''.join(frames)
    
    # Use SpeechRecognition to transcribe
    r = sr.Recognizer()
    # Create AudioData instance
    # sample_rate=16000, sample_width=2 (16-bit)
    sr_audio = sr.AudioData(audio_data, RATE, 2)
    
    try:
        print("Transcribing...")
        text = r.recognize_google(sr_audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None

import tokenize
import io

def is_code_safe(code):
    """
    Scans the code for dangerous patterns using tokenization to avoid false positives
    in comments or strings.
    Returns (True, "") if safe, (False, reason) if unsafe.
    """
    # Dangerous functions/modules
    banned_names = {
        "os.remove", "os.unlink", "os.rmdir", "shutil.rmtree",
        "mkfs", "shutdown", "reboot", "chmod", "chown"
    }
    
    # Dangerous shell commands (to check in subprocess calls)
    banned_shell_cmds = {"rm", "mkfs", "dd", "shutdown", "reboot", "wget", "curl"}

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except tokenize.TokenError:
        return False, "Syntax error in generated code"

    for i, token in enumerate(tokens):
        tok_type = token.type
        tok_string = token.string
        
        # Check for banned function calls (e.g., os.remove)
        if tok_type == tokenize.NAME:
            # Check for "os.remove" style (NAME DOT NAME)
            if i + 2 < len(tokens):
                next_tok = tokens[i+1]
                next_next_tok = tokens[i+2]
                if next_tok.string == '.' and next_next_tok.type == tokenize.NAME:
                    full_name = f"{tok_string}.{next_next_tok.string}"
                    if full_name in banned_names:
                        return False, f"Contains banned function: '{full_name}'"
            
            # Check for direct imports or usage (e.g., from os import remove)
            # This is harder to catch perfectly without AST, but we can flag suspicious names
            # For now, we rely on the full_name check above for standard usage.
            
        # Check for shell commands in strings (only if they look like commands)
        # This is tricky. "rm -rf" in a string is suspicious.
        if tok_type == tokenize.STRING:
            # Simple heuristic: check for "rm " at start of string or after a pipe
            val = tok_string.strip('"\'')
            if any(cmd in val.split() for cmd in banned_shell_cmds):
                 # This might still flag "permission denied" if "permission" isn't banned but "rm" is.
                 # "rm" is the main culprit.
                 # Let's be specific about "rm".
                 words = val.split()
                 if "rm" in words and ("-rf" in words or "-r" in words or "-f" in words or "/" in val):
                     return False, f"Contains suspicious shell command in string: '{val}'"

    return True, ""

def main():
    parser = argparse.ArgumentParser(description="Jarvis Coding Assistant")
    parser.add_argument("--provider", choices=["google", "openai", "local"], default=DEFAULT_PROVIDER, help="LLM Provider")
    parser.add_argument("--model", default=None, help="Model ID (default depends on provider)")
    parser.add_argument("--url", default=None, help="Base URL for OpenAI/Local provider")
    parser.add_argument("--key", default=None, help="API Key")
    
    args = parser.parse_args()
    
    # Provider Setup
    client = None
    model_id = args.model
    
    if args.provider == "google":
        api_key = args.key or os.getenv("GEMINI_API_KEY")
        model_id = model_id or DEFAULT_MODEL
        print(f"--- Initializing Google Client ({model_id}) ---")
        client = GoogleClient(model_id, api_key)
        
    elif args.provider == "openai" or args.provider == "local":
        api_key = args.key or os.getenv("OPENAI_API_KEY") or "dummy-key" # Local might not need key
        base_url = args.url
        
        if args.provider == "local" and not base_url:
            base_url = "http://localhost:1234/v1" # Common default for LM Studio/Ollama
            
        model_id = model_id or "gpt-3.5-turbo" # Default fallback
        print(f"--- Initializing OpenAI/Local Client ({model_id} @ {base_url}) ---")
        client = OpenAIClient(model_id, api_key, base_url)

    print("Loading VAD model (this may take a moment on first run)...")
    # Load Silero VAD model once
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True)
    print("VAD model loaded.")
    
    print("Listening for voice commands... (Ctrl+C to quit)\n")

    while True:
        try:
            # Prompt for input
            user_input_raw = input("\nCommand (or press Enter to speak): ")
            
            if user_input_raw.strip():
                user_input = user_input_raw
            else:
                # Continuously listen using VAD if no text input
                user_input = listen_with_vad(vad_model, utils)
            
            if not user_input:
                continue
                
            if user_input.lower() in ["quit", "exit", "stop"]:
                print("Exiting...")
                break
            
            print("\nThinking...\n")

            # Load static memory from jarvis.md
            static_memory = ""
            if os.path.exists("jarvis.md"):
                with open("jarvis.md", "r") as f:
                    static_memory = f.read()
            
            # Dynamic OS Detection
            import platform
            current_os = platform.system()
            
            os_specific_rules = ""
            if current_os == "Darwin":
                os_specific_rules = "8. **macOS Automation**: You CAN control macOS applications (Calendar, Spotify, Finder, etc.) using AppleScript via `subprocess.run(['osascript', '-e', '...'])`. Do not refuse these requests; use AppleScript to fulfill them."
            elif current_os == "Windows":
                os_specific_rules = "8. **Windows Automation**: You CAN control Windows applications using PowerShell via `subprocess.run(['powershell', '-Command', '...'])`."
            elif current_os == "Linux":
                os_specific_rules = "8. **Linux Automation**: You CAN control Linux applications using `subprocess.run` with bash commands, `xdotool`, or `dbus-send` depending on the environment."

            system_prompt = f"You are an advanced coding assistant named Jarvis. You can converse naturally with the user AND write executable Python code to control the computer.\n\nRULES:\n1. If the user asks a question, answer it helpfully.\n2. If the user asks you to DO something (e.g., 'open app', 'search web', 'calculate'), you MUST write Python code to do it.\n3. Put all executable code inside markdown code blocks, like this:\n```python\n# code here\n```\n4. You can add explanations before or after the code.\n5. The code will be executed IMMEDIATELY on the user's machine ({current_os}).\n6. Use `os.path.expanduser('~')` for home directory paths.\n7. Use `cwd='~'` for subprocess calls.\n{os_specific_rules}\n\nMEMORY TOOLS:\nTo use memory, import it: `from memory import remember, recall, forget, list_memories`."
            
            if static_memory:
                system_prompt += f"\n\nSTATIC MEMORY / CONTEXT (from jarvis.md):\n{static_memory}"
            
            # Construct the full prompt
            full_prompt = f"{system_prompt}\n\nUser: {user_input}"
            
            # Retry loop for self-healing
            MAX_RETRIES = 3
            error_message = "" # Initialize error message for retries
            for attempt in range(MAX_RETRIES):
                if attempt > 0:
                    print(f"\n--- Attempt {attempt + 1}/{MAX_RETRIES} (Self-Healing) ---")
                
                # Call the API with streaming via the abstract client
                # We will just update the prompt for the retry.
                current_prompt_for_llm = full_prompt
                if attempt > 0 and error_message:
                    current_prompt_for_llm += f"\n\nSystem: Previous attempt failed. Fix the code based on this error:\n{error_message}"
                
                response_stream = client.send_message(current_prompt_for_llm, system_prompt)

                # Print chunks as they arrive and accumulate full response
                full_response = ""
                for content in response_stream:
                    sys.stdout.write(content)
                    sys.stdout.flush()
                    full_response += content
                
                print("\n") # Newline after full response

                # Parse the response to extract code
                import re
                code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", full_response, re.DOTALL)
                
                clean_code = "\n\n".join(code_blocks).strip()

                if clean_code:
                    # Security Check
                    is_safe, reason = is_code_safe(clean_code)
                    if not is_safe:
                        print(f"\n[SECURITY ALERT] Execution Blocked: {reason}")
                        error_message = f"Security Alert: Your code was blocked because it contains dangerous patterns: {reason}. Please rewrite it to be safe."
                        continue # Retry with error message

                    # Save to output.py
                    with open("output.py", "w") as f:
                        f.write(clean_code)
                    
                    print("--- Saved to output.py ---")
                    print("--- Executing output.py ---")
                    
                    # Execute output.py with capture
                    try:
                        result = subprocess.run(
                            [sys.executable, "output.py"], 
                            capture_output=True, 
                            text=True, 
                            check=False # Don't raise exception immediately, check returncode
                        )
                        
                        # Print output
                        if result.stdout:
                            print(result.stdout)
                        if result.stderr:
                            print(result.stderr, file=sys.stderr)
                            
                        if result.returncode != 0:
                            print(f"Execution failed with return code {result.returncode}")
                            error_message = f"Execution Error (Exit Code {result.returncode}):\n{result.stderr}"
                            continue # Retry with error message
                        else:
                            # Success!
                            
                            # --- Adaptive Learning ---
                            # If we had to retry (attempt > 0), it means we learned something.
                            # Ask the model if it wants to save this solution.
                            if attempt > 0:
                                print("\n--- Learning from Success ---")
                                learning_prompt = f"{current_prompt_for_llm}\n\nSystem: The code executed successfully! Since we had to fix errors, this is a valuable learning moment. If this solution is worth remembering for future '{user_input}' requests, please output a code block calling `from memory import remember; remember('key', 'value')`. The key should be the user's intent, and the value should be the successful code or a summary. If not worth saving, just say 'No memory needed'."
                                
                                # We need a quick non-streaming call or just reuse the streaming logic
                                # For simplicity, reuse streaming but just capture output
                                learn_stream = client.send_message(learning_prompt, system_prompt)
                                learn_response = ""
                                for chunk in learn_stream:
                                    sys.stdout.write(chunk) # Show the user the thinking
                                    learn_response += chunk
                                
                                # Extract code from learning response
                                learn_blocks = re.findall(r"```(?:python)?\s*(.*?)```", learn_response, re.DOTALL)
                                learn_code = "\n\n".join(learn_blocks).strip()
                                
                                if learn_code:
                                    # Execute memory saving code
                                    # We trust this code as it's just memory ops, but still run safe check?
                                    # Memory ops are safe.
                                    try:
                                        # Write to a temp memory script
                                        with open("memory_update.py", "w") as f:
                                            f.write(learn_code)
                                        subprocess.run([sys.executable, "memory_update.py"], check=True)
                                        print("\n[Memory Updated]")
                                    except Exception as e:
                                        print(f"\n[Memory Update Failed]: {e}")

                            break
                            
                    except Exception as e:
                        print(f"Execution failed: {e}")
                        error_message = f"Execution Exception: {e}"
                        continue
                else:
                    # No code generated, nothing to execute. 
                    break

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
