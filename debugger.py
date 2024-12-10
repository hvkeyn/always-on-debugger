import re
from typing import Dict, Any, Optional
from llm import initialize_llm, Conversation
import logging

# Set up a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class Debugger:
    DELIMITER = "#######"
    FILE_DETECTION_SYSTEM_PROMPT = f"""
    You are an experienced engineer who is helping a junior engineer to debug the error.

    {DELIMITER} Instruction:
    - Carefully read the command run by the user and the resulting error.
    - Identify if the error is related to a specific file. If so, return the file path. If not, return NO_FILE.

    {DELIMITER} Output:
    <filepath>
    {{Just return the file path as a string. If not related to any file, return NO_FILE. Do not add any other information.}}
    </filepath>
    """

    DEBUG_SYSTEM_PROMPT = f"""
    You are an experienced engineer who is helping a junior engineer to debug the error.

    {DELIMITER} Instruction:
    - Carefully read the user's input that includes error and optionally code snippet.
    - Generate a list of hypotheses to explain the error.
    - Then suggest clear and concise steps users can follow to validate the error.

    {DELIMITER}: Output
    <thinking>
    {{Present your thinking and approach here}}
    </thinking>
    <recommendation>
    {{Present your recommendation here. Be very specific, concise and actionable.}}
    </recommendation>
    """

    def __init__(self):
        self.llm = initialize_llm()

    def generate_user_prompt_for_file_detection(self, command: str, error: str) -> str:
        return f"""
        Here is the command and error message:
        ===============
        <command>
        {command}
        </command>
        <error>
        {error}
        </error>
        """

    def detect_file_path(self, command: str, error_message: str) -> Optional[str]:
        logger.debug(f"Error message received: {error_message}")
        match = re.search(r'File \"(.*?)\", line \d+', error_message)
        if match:
            return match.group(1)
        logger.info("No file path detected in the error message.")
        return None

    def generate_user_prompt_for_debug(self, command: str, error: str, code_snippet: Optional[str]) -> str:
        if code_snippet is None:
            return f"""
            Here is the command and error message:
            ===============
            <command>
            {command}
            </command>
            <error>
            {error}
            </error>
            """
        else:
            return f"""
            Here is the command, error message and relevant code:
            ===============
            <command>
            {command}
            </command>
            <error>
            {error}
            </error>
            <code_snippet>
            {code_snippet}
            </code_snippet>
            ===============
            """

    def debug(self, command: str, error_message: str, code_snippet: Optional[str]) -> Dict:
        logger.info("Starting debugging process...")
        messages = [
            {"role": "user", "content": f"Command:\n{command}\n\nError:\n{error_message}"}
        ]
        if code_snippet:
            messages.append({"role": "user", "content": f"Code snippet:\n{code_snippet}"})

        response = self.llm.generate_conversation(messages)
        return {
            "recommendation": response,
            "error_analysis": "Error analysis completed.",
        }

    @staticmethod
    def parse_response(llm_response: str) -> Dict[str, Any]:
        """
        Parse an XML-like string LLM response to get structured output using regex.
        """
        try:
            result = {}

            patterns = {
                'thinking': r'<thinking>(.*?)</thinking>',
                'recommendation': r'<recommendation>(.*?)</recommendation>',
                'filepath': r'<filepath>(.*?)</filepath>'
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, llm_response, re.DOTALL)
                result[key] = match.group(1).strip() if match else None

            return result
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {}