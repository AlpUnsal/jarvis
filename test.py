import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv
import subprocess
import speech_recognition as sr
import torch
import numpy as np
import pyaudio

load_dotenv()

# ---------------- CONFIGURATION ----------------
# Google Gemini API (1500 free requests/day)
# Get your free key from: https://aistudio.google.com/app/apikey
# Add to .env: export GEMINI_API_KEY='AIza...'
API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=API_KEY)

# Google's Flash model (Fast & Free)
MODEL_ID = "gemini-flash-latest"
# -----------------------------------------------

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

def query_coding_model():
    # Initialize the model
    model = genai.GenerativeModel(MODEL_ID)

    print(f"\n--- Connected to {MODEL_ID} ---")
    
    print("Loading VAD model (this may take a moment on first run)...")
    # Load Silero VAD model once
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True)
    print("VAD model loaded.")
    
    print("Listening for voice commands... (Ctrl+C to quit)\n")

    # Initialize chat history
    chat = model.start_chat(history=[])

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
            
            system_prompt = "ONLY RESPOND WITH CODE. YOUR OUTPUT IS GOING TO BE RUN IN THE USER'S LAPTOP IMMEDIATELY AFTER YOU WRITE IT. When the user asks you a question, break down the question into smaller sub-problems. With these sub-problems, define DRY functions to use as building blocks. Then, implement the functions in a clean, efficient way. If the user asks you to open a website, write code to open the website in python. IMPORTANT: When using file paths, ALWAYS use os.path.expanduser('~') to resolve the home directory. NEVER use '~' directly in paths. You are operating the laptop of the user. Every script you write is being run on the user's laptop, it is a macbook pro with the M1 chip. Always run the script in the user's home directory using cwd='~'.\n\nMEMORY TOOLS:\nYou have access to a persistent memory module. To use it, you MUST import it in your generated code: `from memory import remember, recall, forget, list_memories`.\n- Use `remember(key, value)` to store information.\n- Use `recall(key)` to retrieve information.\n- Use `forget(key)` to delete information.\n- Use `list_memories()` to see what is stored.\nUse these tools when the user asks you to remember something or recall something you've been told before."
            
            if static_memory:
                system_prompt += f"\n\nSTATIC MEMORY / CONTEXT (from jarvis.md):\n{static_memory}"

            # Construct the full prompt
            full_prompt = f"{system_prompt}\n\nUser: {user_input}"

            # Call the API with streaming
            response_stream = chat.send_message(full_prompt, stream=True)

            # Print chunks as they arrive and accumulate full response
            full_response = ""
            for chunk in response_stream:
                if chunk.text:
                    content = chunk.text
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

            if clean_code:
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
            else:
                print("No code generated.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    query_coding_model()
