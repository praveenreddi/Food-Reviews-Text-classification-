import os
import json
import requests
from typing import Dict, List, Any, Optional
import autogen
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

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

class CustomCallbackHandler:
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM started with prompts: {prompts}")

    def on_llm_end(self, response, **kwargs):
        print(f"LLM ended with response: {response}")

    def on_llm_error(self, error, **kwargs):
        print(f"LLM error: {error}")

# Initialize your LLM instance
llm_instance = LlamaLLM(llm_model="openai")  # Using openai for gpt-4-32k

# Create a custom completion function for AutoGen
def custom_completion(messages, model="gpt-4-32k", temperature=0, max_tokens=800):
    # Format the conversation history for your LLM
    # Extract the last user message
    last_message = None
    for msg in reversed(messages):
        if msg["role"] == "user":
            last_message = msg["content"]
            break

    if not last_message:
        last_message = messages[-1]["content"]

    # Call your existing _call method to get the response
    response = llm_instance._call(last_message)

    # Return in the format AutoGen expects
    return {"choices": [{"message": {"content": response, "role": "assistant"}}]}

# Configure AutoGen with your custom completion function
config_list = [
    {
        "model": "gpt-4-32k",
        "api_key": "not-needed"
    }
]

# Create AutoGen agents with the custom completion function
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": config_list,
        "cache_seed": 42,
        "temperature": 0,
        "timeout": 120
    },
    system_message="You are a helpful AI assistant."
)

# Register the custom completion function
autogen.ChatCompletion.register_provider(
    model_type="gpt-4-32k",
    provider="custom",
    completion_constructor=custom_completion
)

# Set the provider for the assistant
assistant.llm_config["provider"] = "custom"

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
