from core.llm import LLMClient
from core.executor import CodeExecutor

def main():
    llm = LLMClient("mlx-community/Qwen2.5-Coder-7B-4bit")
    executor = CodeExecutor()

    while True:
        user_input = input("Enter a command: ")
        response = llm.send_message(user_input, "You are a helpful assistant, respond to the user.")
        print(response)

if __name__ == "__main__":
    main()