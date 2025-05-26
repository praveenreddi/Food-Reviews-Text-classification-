import requests
import autogen
from autogen.agentchat.contrib.llm_compiler import LLMCompiler

# Your custom LLM class
class LlamaLlu:
    def __init__(self, llm_model="openai"):
        self.llm_model = llm_model
        if llm_model == "openai":
            self.endpoint = "https://api.openai.com/v1/chat/completions"
            self.model = "gpt-3.5-turbo"
        elif llm_model == "llama":
            self.endpoint = "http://localhost:8000/v1/chat/completions"
            self.model = "llama-2"
        else:
            raise ValueError("Unknown model")
        self.access_token = "YOUR_ACCESS_TOKEN"  # <-- Replace with your actual token

    def __call__(self, prompt):
        print(f"Calling {self.llm_model} with prompt: {prompt}")
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        response = requests.post(self.endpoint, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            print("Error:", response.text)
            return "Error: LLM call failed"

# AutoGen wrapper for your LLM
class CustomLLMConfig(LLMCompiler):
    def __init__(self, custom_llm, **kwargs):
        super().__init__(**kwargs)
        self.custom_llm = custom_llm

    def create_completion(self, messages, **kwargs):
        prompt = self._messages_to_prompt(messages)
        response = self.custom_llm(prompt)
        return {"choices": [{"message": {"content": response}}]}

    def _messages_to_prompt(self, messages):
        prompt = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            prompt += f"{role}: {content}\n"
        return prompt

# Instantiate your LLM
my_llm = LlamaLlu(llm_model="openai")  # or "llama" if you want

# Create the custom config for AutoGen
custom_llm_config = {
    "config_list": [{"model": "custom-model"}],
    "custom_llm": CustomLLMConfig(my_llm)
}

# Set up AutoGen agents
assistant = autogen.AssistantAgent(
    name="Assistant",
    llm_config=custom_llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10
)

# Start the conversation
if __name__ == "__main__":
    user_proxy.initiate_chat(
        assistant,
        message="What is the capital of India?"
    )
