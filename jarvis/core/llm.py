from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

class LLMClient:
    def __init__(self, model_name: str, max_tokens: int = 1000, temp: float = 0.7, top_p: float = 0.95, xtc_threshold: float = 0.0, xtc_probability: float = 0.0, eos_token_ids: list = [151645]):
        """
        Initialize the LLM client with the given model name.
        Args:
            model_name: The name of the model to use.
            max_tokens: The maximum number of tokens to generate.
            temp: The temperature of the model.
            top_p: The top-p value of the model.
            xtc_threshold: The threshold for the model to use XTC.
            xtc_probability: The probability for the model to use XTC.
            eos_token_ids: The EOS token IDs to use.
        """
        self.model_name = model_name
        self.model, self.tokenizer = load(model_name)
        
        # qwen models use 151645 (<|im_end|>) as eos but tokenizer only has 151643
        if hasattr(self.tokenizer, 'eos_token_ids'):
            self.tokenizer.eos_token_ids.add(151645)

        self.prompt_cache = make_prompt_cache(self.model)
        self.conversation_history = []
        self.max_tokens = max_tokens
        self.temp = temp
        self.top_p = top_p
        self.stop_token_ids = {151643, 151645}

    def send_message(self, user_input: str, system_prompt: str) -> str:
        """
        Send a message to the LLM and return the response.
        Args:
            user_input: The user's input.
            system_prompt: The system prompt.
        """

        prompt = self.make_prompt(user_input, system_prompt)
        
        response = None
        full_response = ""
        consecutive_empty = 0
        max_consecutive_empty = 5
        
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=make_sampler(
                self.temp,
                self.top_p,
            ),
            prompt_cache=self.prompt_cache
        ):
            
            print(response.text, flush=True, end="")
            full_response += response.text

        print()
        self.conversation_history.append({"role": "user", "content": user_input})
        return full_response


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
        
        return self.tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
        )

    def reset_chat(self):
        """
        Reset the chat history.
        """
        self.conversation_history = []
        self.prompt_cache = make_prompt_cache(self.model)
