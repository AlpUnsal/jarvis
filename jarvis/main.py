from core.llm import LLMClient
from core.executor import CodeExecutor

def print_help():
    print("The command list:")
    print("- 'q' to exit")
    print("- 'r' to reset the chat")
    print("- 'h' to display these commands")


def main():
    llm = LLMClient(model_name="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    executor = CodeExecutor()

    print(f"[INFO] Starting chat session with {llm.model_name}.")
    print_help()
    while True:
        user_input = input(">> ")
        if user_input == 'q':
            break
        elif user_input == 'r':
            llm.reset_chat()
        elif user_input == 'h':
            print_help()
        else:
            response = llm.send_message(user_input, "You are an advanced coding assistant named Jarvis. Respond to the user with only executable python code. The user is on Mac OS.")
            #print(response)
            '''
            result = executor.execute_code(response)
            if 'error' in result:
                print(f"Error: {result['error']}")
                if 'traceback' in result:
                    print(result['traceback'])
            else:
                print(result.get('output', ''))
                print(result.get('variables', {}))
            '''

if __name__ == "__main__":
    main()