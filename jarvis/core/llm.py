from mlx_lm import load, generate
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache

class LLMClient:
    def __init__(self, model_name: str):
        """
        Initialize the LLM client with the given model name.
        Args:
            model_name: The name of the model to use.
        """
        self.model, self.tokenizer = load(model_name)
        self.prompt_cache = make_prompt_cache(self.model)
        self.conversation_history = []

    def send_message(self, user_input: str, system_prompt: str) -> str:
        """
        Send a message to the LLM and return the response.
        Args:
            user_input: The user's input.
            system_prompt: The system prompt.
        """

        prompt = self.make_prompt(user_input, system_prompt)
        
        # Get prompt length in tokens for extraction
        prompt_tokens = prompt if isinstance(prompt, list) else self.tokenizer.encode(prompt)
        prompt_length = len(prompt_tokens)

        full_response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            verbose=False,  # Turn off verbose to avoid duplicate output
            prompt_cache=self.prompt_cache,
        )
        
        # Extract only the newly generated part by tokenizing and taking tokens after prompt
        if full_response:
            full_tokens = self.tokenizer.encode(full_response)

            if len(full_tokens) >= prompt_length and full_tokens[:prompt_length] == prompt_tokens:
                new_tokens = full_tokens[prompt_length:]
                response = self.tokenizer.decode(new_tokens).strip()
            else:
                # Fallback to string-based extraction if token matching fails
                prompt_text = self.tokenizer.decode(prompt_tokens)
                if full_response.startswith(prompt_text):
                    response = full_response[len(prompt_text):].strip()
                else:
                    # If string matching also fails, return the full response
                    response = full_response.strip()
        else:
            response = ""
        
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response


    def make_prompt(self, user_input: str, system_prompt: str = "") -> str:
        """
        Make a prompt for the LLM including full conversation history.
        Args:
            user_input: The user's input.
            system_prompt: The system prompt (optional).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.extend(self.conversation_history)

        messages.append({"role": "user", "content": user_input})
        
        prompt = self.tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
        )

        return prompt