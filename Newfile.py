import os
import json
import requests
from typing import Dict, List, Any, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

# Install required packages if needed
try:
    import autogen
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "pyautogen"])
    import autogen

class LlamaLLM(LLM):
    llm_model: str
    llm_url: str = None
    model: str = None

    def __init__(self, llm_model="llama"):
        super().__init__()
        self.llm_model = llm_model

        if llm_model == "llama":
            self.llm_url = "https://api.abacus.ai/api/v0/llm/completions"
            self.model = "llama3-70b-instruct"
        elif llm_model == "openai":
            self.llm_url = "https://api.abacus.ai/api/v0/llm/completions"
            self.model = "gpt-4-32k"

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        access_token = get_access_token()

        messages = [{"role": "user", "content": prompt}]

        response = call_openai(
            messages=messages,
            llm_url=self.llm_url,
            access_token=access_token,
            model=self.model,
            temperature=0,
            max_tokens=800,
            top_p=1
        )

        return response

def get_access_token():
    # Replace with your actual token retrieval logic
    return os.environ.get("ABACUS_API_TOKEN", "your-default-token")

def call_openai(messages, llm_url, access_token, model, temperature=0, max_tokens=800, top_p=1):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p
    }

    response = requests.post(llm_url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        raise Exception(f"API call failed with status code {response.status_code}: {response.text}")

# Create a custom endpoint class that will be used by AutoGen
class CustomEndpoint:
    def __init__(self):
        self.llm = LlamaLLM(llm_model="openai")  # Using openai for gpt-4-32k

    def completion(self, messages, model=None, temperature=0, max_tokens=None, stream=False):
        """Custom completion function that mimics OpenAI's API"""
        if stream:
            raise ValueError("Streaming is not supported with this custom endpoint")

        # Extract the conversation context
        prompt = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            prompt += f"{role}: {content}\n"

        # Get response from your LLM
        response_text = self.llm._call(prompt)

        # Return in the format expected by AutoGen
        return {
            "choices": [
                {
                    "message": {
                        "content": response_text,
                        "role": "assistant"
                    }
                }
            ]
        }

# Create a custom implementation of the OpenAI client
class CustomOpenAI:
    def __init__(self, api_key=None, base_url=None, **kwargs):
        self.endpoint = CustomEndpoint()

    def chat(self):
        return self.endpoint

# Monkey patch AutoGen to use our custom OpenAI client
import importlib
import sys

# Create a module for our custom OpenAI client
custom_openai_module = type(sys)(name='custom_openai')
custom_openai_module.Client = CustomOpenAI
sys.modules['custom_openai'] = custom_openai_module

# Patch AutoGen to use our custom module
autogen.oai.openai = custom_openai_module

# Set up the agents
config_list = [
    {
        "model": "gpt-4",
        "api_key": "sk-dummy-key-for-custom-implementation"
    }
]

# Create AutoGen agents
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": config_list,
        "temperature": 0
    },
    system_message="You are a helpful AI assistant."
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "coding"}
)

# Start the conversation
user_proxy.initiate_chat(
    assistant,
    message="Help me solve this problem: calculate the factorial of 5."
)
