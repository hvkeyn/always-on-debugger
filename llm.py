# Related third-party imports
import abc
import os
import logging
import requests
import json
from dotenv import load_dotenv, find_dotenv
from typing import Callable, Dict, List, Optional

import openai
import anthropic

# Set up a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Environment setup
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

MODEL_MAPPING = {
    "openai": {
        "small": "gpt-3.5-turbo-0125",
        "medium": "gpt-4",
        "large": "gpt-4-turbo",
        "latest_large": "gpt-4o"
    },
    "anthropic": {
        "small": "claude-3-haiku-20240307",
        "medium": "claude-3-sonnet-20240229",
        "large": "claude-3-opus-20240229",
        "latest_large": "claude-3-5-sonnet-20240620"
    },
    "local": {
        "embedding": "text-embedding-nomic-embed-text-v1.5"
    }
}

class LLM(abc.ABC):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    @abc.abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        pass

    @abc.abstractmethod
    def stream_text(self, prompt: str, **kwargs):
        pass

    def embed_text(self, input_texts: List[str], model: str = "embedding") -> List[Dict]:
        raise NotImplementedError("This LLM does not support embeddings.")

class OpenAILLM(LLM):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.environ.get("OPENAI_API_KEY"))
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate_text(self, prompt: str, model: str = "large", **kwargs) -> str:
        model_name = MODEL_MAPPING["openai"][model]
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0.2),
        )
        logger.info(f"Response from OpenAI: {response.choices[0].message.content}")
        return response.choices[0].message.content

    def stream_text(self, prompt: str, model: str = "large", **kwargs):
        model_name = MODEL_MAPPING["openai"][model]
        messages = [{"role": "user", "content": prompt}]
        stream = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0.2),
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].message['content'] is not None:
                yield chunk.choices[0].message['content']

    def generate_conversation(self, messages, model: str = "large", **kwargs) -> str:
        model_name = MODEL_MAPPING["openai"][model]
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0.2),
        )
        logger.info(f"Response from OpenAI: {response.choices[0].message.content}")
        return response.choices[0].message.content

    def generate_conversation_stream(self, messages, model: str = "large", **kwargs):
        model_name = MODEL_MAPPING["openai"][model]
        stream = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=kwargs.get("temperature", 0.2),
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response_string = chunk.choices[0].delta.content
                yield response_string

class AnthropicLLM(LLM):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or ANTHROPIC_API_KEY)
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate_text(self, prompt: str, model: str = "large", **kwargs) -> str:
        model_name = MODEL_MAPPING["anthropic"][model]
        response = self.client.messages.create(
            model=model_name,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 4096),
            messages=[{"role": "user", "content": prompt}]
        )
        logger.info(f"Response from Anthropic: {response.content[0].text}")
        return response.content[0].text

    def generate_conversation(self, system_message, messages, model: str = "large", **kwargs) -> str:
        model_name = MODEL_MAPPING["anthropic"][model]
        system = system_message
        response = self.client.messages.create(
            model=model_name,
            system=system,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 4096),
            messages=messages
        )
        return response.content[0].text

    def stream_text(self, prompt: str, model: str = "large", **kwargs):
        model_name = MODEL_MAPPING["anthropic"][model]
        messages = [{"role": "user", "content": prompt}]
        with self.client.messages.stream(
            model=model_name,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 4096),
            messages=messages,
        ) as stream:
            for chunk in stream.text_stream:
                yield chunk

    def generate_conversation_stream(self, system_message, messages, model: str = "large", **kwargs):
        model_name = MODEL_MAPPING["anthropic"][model]
        system = system_message
        with self.client.messages.stream(
            system=system,
            model=model_name,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 4096),
            messages=messages,
        ) as stream:
            for chunk in stream.text_stream:
                if chunk is not None:
                    yield chunk

    def generate_conversation_stream_print(self, system_message, messages, model: str = "large", **kwargs):
        model_name = MODEL_MAPPING["anthropic"][model]
        system = system_message
        response = ""
        with self.client.messages.stream(
            system=system,
            model=model_name,
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 4096),
            messages=messages,
        ) as stream:
            for chunk in stream.text_stream:
                if chunk is not None:
                    response += chunk
        return response


class LocalEmbeddingLLM(LLM):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key)
        self.endpoint = "http://127.0.0.1:1234/v1/chat/completions"
        self.listpoint = "http://127.0.0.1:1234/v1/models"
        self.chat_completions_url = "http://127.0.0.1:1234/v1/chat/completions"

    def list_models(self) -> List[str]:
        """
        Get a list of available models from the local model server.
        """
        try:
            # Use the models endpoint to fetch available models
            models_endpoint = self.endpoint.replace("/v1/chat/completions", "/v1/models")
            response = requests.get(models_endpoint)

            if response.status_code == 200:
                models_data = response.json()

                # Extract model IDs from the 'data' field
                if "data" in models_data:
                    return [model["id"] for model in models_data["data"]]
                else:
                    logger.error(f"Unexpected response format: {models_data}")
                    return []
            else:
                logger.error(f"Failed to get models: {response.text}")
                return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error while fetching models: {e}")
            return []

    def embed_text(self, input_texts: List[str], model: str = "text-embedding-nomic-embed-text-v1.5") -> List[Dict]:
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model,
            "input": input_texts
        }
        response = requests.post(self.endpoint, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error from local embedding server: {response.text}")
            return []

    def generate_text(self, prompt: str, **kwargs) -> str:
        logger.info("LocalEmbeddingLLM does not support text generation.")
        return "LocalEmbeddingLLM does not support text generation."

    def stream_text(self, prompt: str, **kwargs):
        logger.info("LocalEmbeddingLLM does not support text streaming.")
        yield "LocalEmbeddingLLM does not support text streaming."

    # Handling chat completions
    def generate_conversation(self, messages: List[Dict[str, str]], **kwargs) -> str:
        payload = {"messages": messages, **kwargs}
        try:
            logger.info(f"Sending request to local model: {self.chat_completions_url}")
            logger.debug(f"Request payload: {payload}")

            # Make the request
            response = requests.post(self.chat_completions_url, json=payload)
            response.raise_for_status()

            # Parse the response
            data = response.json()
            if "choices" in data and data["choices"]:
                # Extract the content from the assistant's response
                content = data["choices"][0]["message"]["content"]
                logger.debug(f"Received response content: {content}")
                return content
            else:
                logger.error("No valid choices in the response.")
                return "Error: No response received."
        except Exception as e:
            logger.error(f"Error from local chat completions server: {e}")
            return f"Error: {e}"

    def generate_conversation_stream(self, messages: List[Dict[str, str]], **kwargs):
        payload = {"messages": messages, **kwargs}
        try:
            logger.info(f"Sending request to local model: {self.chat_completions_url}")
            logger.debug(f"Request payload: {payload}")

            # Send request
            response = requests.post(self.chat_completions_url, json=payload, stream=False)
            response.raise_for_status()

            # Yield the full response
            data = response.json()
            if "choices" in data and data["choices"]:
                yield data["choices"][0]["message"]["content"]
            else:
                logger.error("No valid choices in the response.")
                yield "Error: No valid response received."
        except Exception as e:
            logger.error(f"Error from local chat completions server: {e}")
            yield f"Error: {e}"

    def generate_conversation_stream_print(self, system_message: str, messages: List[Dict[str, str]], **kwargs) -> str:
        logger.info("Streaming response from LocalEmbeddingLLM.")
        result = ""
        try:
            response = self.generate_conversation_stream(messages, **kwargs)
            for chunk in response:
                if chunk.strip():
                    result += chunk
                else:
                    logger.warning("Received an empty or invalid chunk.")
        except Exception as e:
            logger.error(f"Error during streaming print: {e}")
            return f"Error occurred during streaming: {e}"
        return result

    def validate_messages(self, messages):  # Add self as the first argument
        if not isinstance(messages, list) or not all(
                isinstance(m, dict) and "role" in m and "content" in m for m in messages):
            logger.error(f"Invalid 'messages' format: {messages}")
            return False
        return True


def initialize_llm():
    print("initialize_llm...")

    # Check for OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        print("OpenAILLM!")
        return OpenAILLM(api_key=openai_api_key)

    # Check for Anthropic API key
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        print("AnthropicLLM!")
        return AnthropicLLM(api_key=anthropic_api_key)

    # Default to local model
    print("LocalEmbeddingLLM!")
    return LocalEmbeddingLLM()

class Message:
    def __init__(
        self,
        role: str,
        content: str,
        name: str = None,
    ):
        self.role = role
        self.content = content
        self.name = name

    def to_dict(self):
        message_dict = {"role": self.role, "content": self.content}
        if self.name:
            message_dict["name"] = self.name
        return message_dict

class Conversation:
    def __init__(self, system_message=None):
        self.messages = []
        if system_message:
            self.messages.append(Message(role="system", content=system_message))

    def __iter__(self):
        for message in self.messages:
            yield message

    def to_dict(self):
        return [message.to_dict() for message in self.messages]

    def add_message(self, message: Dict[str, str]):
        self.messages.append(Message(**message))

    def add_user_message(self, question: str):
        self.add_message({"role": "user", "content": question})

    def add_assistant_message(self, answer: str):
        self.add_message({"role": "assistant", "content": answer})

    def print_conversation(self):
        for message in self.messages:
            logger.info(f"{message.role}: {message.content}")