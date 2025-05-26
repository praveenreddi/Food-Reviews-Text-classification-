import requests
import autogen

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

# Instantiate your LLM
my_llm = LlamaLlu(llm_model="openai")  # or "llama"

# Define a completion function for AutoGen
def my_completion_fn(messages, **kwargs):
    prompt = ""
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        prompt += f"{role}: {content}\n"
    return {"choices": [{"message": {"content": my_llm(prompt)}}]}

# LLM config for AutoGen
llm_config = {
    "completion_fn": my_completion_fn,
    "config_list": [{"model": "custom-llm"}],
}

# Set up AutoGen agents
assistant = autogen.AssistantAgent(
    name="Assistant",
    llm_config=llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=3
)

# Start the conversation
if __name__ == "__main__":
    user_proxy.initiate_chat(
        assistant,
        message="What is the capital of India?"
    )

# Additional example: Multi-agent conversation
def create_multi_agent_example():
    """
    Example of creating multiple agents for a more complex conversation
    """
    # Create a researcher agent
    researcher = autogen.AssistantAgent(
        name="Researcher",
        system_message="You are a helpful research assistant. You provide detailed, accurate information.",
        llm_config=llm_config
    )

    # Create a critic agent
    critic = autogen.AssistantAgent(
        name="Critic",
        system_message="You are a critical thinker. You question and verify information provided by others.",
        llm_config=llm_config
    )

    # Create a user proxy for multi-agent
    user_proxy_multi = autogen.UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    )

    return researcher, critic, user_proxy_multi

# Uncomment the lines below to run the multi-agent example
# researcher, critic, user_proxy_multi = create_multi_agent_example()
# user_proxy_multi.initiate_chat(
#     researcher,
#     message="Research the benefits of renewable energy and provide a summary."
# )

