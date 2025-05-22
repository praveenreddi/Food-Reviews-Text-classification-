import os
import json
import requests
from typing import Dict, List, Any, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

# Install required packages
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

# Create a custom completion function for AutoGen
def custom_completion(messages, model=None, temperature=0, max_tokens=None, **kwargs):
    """Custom completion function that uses your LlamaLLM"""
    llm = LlamaLLM(llm_model="openai")  # Using openai for gpt-4-32k

    # Format the messages for your API
    prompt = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        prompt += f"{role}: {content}\n"

    # Get response from your LLM
    response_text = llm._call(prompt)

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

# Register the custom completion function with AutoGen
autogen.ChatCompletion.register_completion(
    custom_completion,
    model_type="custom",
    api_key="not-needed"
)

# Create a config list with your custom model
config_list = [
    {
        "model": "custom",
        "api_key": "not-needed"
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
