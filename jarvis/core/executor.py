import traceback
import contextlib
import io
import re

class CodeExecutor:
    def __init__(self):
        """
        Initialize the code executor.
        """
        self.output_buffer = io.StringIO()
        self.local_scope = {}

    def extract_code(self, response: str) -> str:
        """
        Extract code from markdown code blocks in the LLM response.
        """
        
        pattern = r'```(?:python|py)?\s*\n(.*?)```'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # if no code block found, return the original response
        return response.strip()

    def execute_code(self, code: str) -> str:
        """
        Execute the given code and return the result.
        """
        code = self.extract_code(code)
        try:
            with contextlib.redirect_stdout(self.output_buffer):
                exec(code, {}, self.local_scope)
            output = self.output_buffer.getvalue()
            return {"output": output, "variables": self.local_scope}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}

    def get_output(self) -> str:
        """
        Get the output of the last code execution.
        """
        return self.output_buffer.getvalue()